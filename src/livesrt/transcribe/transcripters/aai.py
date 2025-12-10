"""
AssemblyAI implementation of the Transcripter interface.
"""

import asyncio
import json
import logging
from asyncio import CancelledError
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import httpx
import websockets

from ..base import AudioSource, Transcripter, TranscriptReceiver, Turn, Word

logger = logging.getLogger(__name__)


@dataclass
class AssemblyAITranscripter(Transcripter):
    """
    Transcribes audio using the AssemblyAI Streaming API.
    """

    api_key: str
    region: Literal["eu", "us"] = "eu"
    end_of_turn_confidence_threshold: float = 0.4
    format_turns: bool = True
    inactivity_timeout: timedelta | None = None
    keyterms_prompt: list[str] = field(default_factory=list)
    language_detection: bool = True
    min_end_of_turn_silence: timedelta = timedelta(milliseconds=400)
    max_turn_silence: timedelta = timedelta(milliseconds=1280)
    speech_model: str = "universal-streaming-multilingual"

    @property
    def domain(self) -> str:
        """Domain name for the streaming API"""
        if self.region == "eu":
            return "streaming.eu.assemblyai.com"
        return "streaming.assemblyai.com"

    @asynccontextmanager
    async def _http_client(self):
        async with httpx.AsyncClient(
            base_url=f"https://{self.domain}/",
            timeout=30,
            headers={"Authorization": self.api_key},
        ) as client:
            yield client

    async def _get_stream_token(self) -> str:
        """Obtains a temporary token for the streaming WebSocket."""
        async with self._http_client() as client:
            resp = await client.get("/v3/token", params=dict(expires_in_seconds=60))
            resp.raise_for_status()
            return resp.json()["token"]

    async def health_check(self) -> None:
        """Checks if the API key is valid by making a lightweight request."""
        try:
            await self._get_stream_token()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                error_msg = "Invalid AssemblyAI API key."
                raise ValueError(error_msg) from e
            raise
        except Exception as e:
            error_msg = f"AssemblyAI connection failed: {e}"
            raise RuntimeError(error_msg) from e

    def get_settings(self) -> dict[str, str]:
        """Returns a dictionary of relevant settings for display."""
        return {
            "Provider": "AssemblyAI",
            "Region": self.region,
            "Model": self.speech_model,
            "Language Detection": str(self.language_detection),
            "End of Turn Threshold": str(self.end_of_turn_confidence_threshold),
        }

    async def process(self, source: AudioSource, receiver: TranscriptReceiver) -> None:
        """
        Main processing loop. connects to AAI, streams audio from source,
        and feeds results to receiver.
        """
        token = await self._get_stream_token()

        # Map generic audio format to AAI specific encoding
        encoding = "pcm_mulaw" if source.get_format() == "ulaw" else "pcm_s16le"

        ws_url = httpx.URL(f"wss://{self.domain}/v3/ws").copy_merge_params(
            dict(
                sample_rate=source.get_sample_rate(),
                encoding=encoding,
                end_of_turn_confidence_threshold=self.end_of_turn_confidence_threshold,
                format_turns="true" if self.format_turns else "false",
                inactivity_timeout=(
                    str(self.inactivity_timeout.total_seconds())
                    if self.inactivity_timeout
                    else None
                ),
                keyterms_prompt=self.keyterms_prompt,
                language_detection="true" if self.language_detection else "false",
                min_end_of_turn_silence_when_confident=str(
                    self.min_end_of_turn_silence.total_seconds() * 1000
                ),
                max_turn_silence=str(self.max_turn_silence.total_seconds() * 1000),
                speech_model=self.speech_model,
                token=token,
            )
        )

        termination_received = asyncio.Event()
        should_send_terminate = asyncio.Event()

        # Notify receiver we are starting
        await receiver.start()

        async with websockets.connect(str(ws_url)) as ws:
            tx_t = asyncio.create_task(
                self._stream_tx(source, should_send_terminate, ws)
            )
            rx_t = asyncio.create_task(
                self._stream_rx(ws, receiver, termination_received)
            )

            try:
                done, _ = await asyncio.wait(
                    [tx_t, rx_t],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Check for exceptions
                for task in done:
                    if not task.cancelled():
                        exc = task.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                should_send_terminate.set()
                raise

            finally:
                # Graceful termination sequence
                if should_send_terminate.is_set() and not termination_received.is_set():
                    try:
                        async with asyncio.timeout(3):
                            await ws.send(json.dumps({"type": "Terminate"}))
                            await termination_received.wait()
                    except (TimeoutError, Exception):
                        logger.warning("Graceful termination failed or timed out")

                # Ensure tasks are cleaned up
                tx_t.done() or tx_t.cancel()
                rx_t.done() or rx_t.cancel()
                await asyncio.gather(tx_t, rx_t, return_exceptions=True)

                # Notify receiver we are done
                await receiver.stop()

    async def _stream_tx(
        self,
        source: AudioSource,
        should_send_terminate: asyncio.Event,
        ws: websockets.ClientConnection,
    ) -> None:
        """
        Transmission loop: Reads from AudioSource iterator and sends to WebSocket.
        """
        try:
            async for chunk in source.iter_frames():
                if not chunk:
                    break
                await ws.send(chunk)

            # Source exhausted (file ended, or mic stopped)
            should_send_terminate.set()

        except asyncio.CancelledError:
            should_send_terminate.set()
            raise
        except Exception:
            should_send_terminate.set()
            logger.exception("Error in transmission loop")
            raise

    async def _stream_rx(
        self,
        ws: websockets.ClientConnection,
        receiver: TranscriptReceiver,
        term_event: asyncio.Event,
    ) -> None:
        """
        Reception loop: Reads JSON from WebSocket and calls Receiver.
        """
        while not term_event.is_set():
            try:
                msg_raw = await ws.recv()
            except websockets.ConnectionClosed:
                break

            try:
                msg = json.loads(msg_raw)
            except json.JSONDecodeError:
                continue

            try:
                await self._handle_message(msg, receiver, term_event)
            except (CancelledError, SystemExit):
                raise
            except Exception:
                logger.exception("Could not handle message")

    async def _handle_message(
        self,
        msg: Any,
        receiver: TranscriptReceiver,
        term_event: asyncio.Event,
    ) -> None:
        match msg:
            case {"type": "SessionBegins"}:
                # We already called receiver.start() in process(), so we log this
                logger.info("AAI Session started: %s", msg.get("session_id"))

            case {"type": "Termination"}:
                term_event.set()

            case {
                "type": "Turn",
                "transcript": transcript,
            }:
                # Parse AAI Turn into Generic Turn
                turn_id = msg.get("turn_order", 0)
                is_final = msg.get("end_of_turn", False)

                # Parse words if available
                words_data = msg.get("words", [])
                words_list = []

                for w in words_data:
                    words_list.append(
                        Word(
                            type="word",
                            text=w.get("text", ""),
                            start=timedelta(milliseconds=w.get("start", 0)),
                            end=timedelta(milliseconds=w.get("end", 0)),
                            speaker=msg.get("speaker", ""),
                        )
                    )

                # Only emit if there is content
                if transcript:
                    turn = Turn(
                        id=turn_id,
                        text=transcript,
                        final=is_final,
                        words=words_list,
                    )
                    await receiver.receive_turn(turn)

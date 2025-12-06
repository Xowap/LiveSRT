"""
ElevenLabs implementation of the Transcripter interface.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
import websockets

from ...errors import TranscribeError
from ..base import AudioSource, Transcripter, TranscriptReceiver, Turn, Word

logger = logging.getLogger(__name__)


@dataclass
class ElevenLabsTranscripter(Transcripter):
    """
    Transcribes audio using the ElevenLabs Realtime Speech-to-Text API.
    """

    api_key: str
    model_id: str = "scribe_v2_realtime"
    commit_strategy: str = "vad"
    vad_silence_threshold_secs: float = 0.8
    include_timestamps: bool = True

    @property
    def url(self) -> str:
        """The WebSocket URL for the ElevenLabs API."""
        return "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    async def process(self, source: AudioSource, receiver: TranscriptReceiver) -> None:
        """
        Main processing loop. Connects to ElevenLabs, streams audio,
        and feeds results to receiver.
        """
        # Determine audio format string for query params
        sample_rate = source.get_sample_rate()
        fmt_code = source.get_format()  # "pcm" or "ulaw"
        audio_format = f"{fmt_code}_{sample_rate}"

        # Construct WebSocket URL with query parameters
        ws_url = httpx.URL(self.url).copy_merge_params(
            dict(
                model_id=self.model_id,
                audio_format=audio_format,
                commit_strategy=self.commit_strategy,
                vad_silence_threshold_secs=str(self.vad_silence_threshold_secs),
                include_timestamps="true" if self.include_timestamps else "false",
                # ElevenLabs accepts the API key in the header 'xi-api-key'
            )
        )

        termination_received = asyncio.Event()
        should_send_terminate = asyncio.Event()
        session_started = asyncio.Event()  # NEW: wait for session to start

        # Notify receiver we are starting
        await receiver.start()

        # Headers for authentication
        extra_headers = {"xi-api-key": self.api_key}

        async with websockets.connect(
            str(ws_url),
            additional_headers=extra_headers,
        ) as ws:
            tx_t = asyncio.create_task(
                self._stream_tx(source, should_send_terminate, session_started, ws)
            )
            rx_t = asyncio.create_task(
                self._stream_rx(ws, receiver, termination_received, session_started)
            )

            try:
                done, _ = await asyncio.wait(
                    [tx_t, rx_t],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Check for exceptions
                for task in done:
                    if not task.cancelled():
                        task.result()

            except asyncio.CancelledError:
                should_send_terminate.set()
                raise

            finally:
                # Graceful shutdown not explicitly defined in EL protocol via a
                # "Terminate" message like AAI, but stopping the audio stream
                # effectively ends it.
                should_send_terminate.set()

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
        session_started: asyncio.Event,
        ws: websockets.ClientConnection,
    ) -> None:
        """
        Transmission loop: Reads from AudioSource, base64 encodes, wraps in JSON
        and sends to WebSocket.
        """
        sample_rate = source.get_sample_rate()

        try:
            # WAIT for session_started message before sending audio
            await session_started.wait()
            logger.info("Session started, beginning audio transmission")

            async for chunk in source.iter_frames():
                if should_send_terminate.is_set():
                    break

                if not chunk:
                    break

                # ElevenLabs expects base64 encoded strings
                b64_audio = base64.b64encode(chunk).decode("utf-8")

                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": b64_audio,
                    "commit": False,  # We let VAD handle commits
                    "sample_rate": sample_rate,
                }

                await ws.send(json.dumps(msg))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in transmission loop")
            raise

    async def _stream_rx(
        self,
        ws: websockets.ClientConnection,
        receiver: TranscriptReceiver,
        term_event: asyncio.Event,
        session_started: asyncio.Event,
    ) -> None:
        """
        Reception loop: Reads JSON from WebSocket and calls Receiver.
        """
        turn_counter = 0

        while not term_event.is_set():
            try:
                msg_raw = await ws.recv()
            except websockets.ConnectionClosed:
                break

            try:
                msg = json.loads(msg_raw)
            except json.JSONDecodeError:
                continue

            # Check for errors first
            if msg.get("message_type", "").endswith("_error"):
                logger.error("ElevenLabs Error: %s", msg)
                # In case of fatal errors (auth, quota), we should probably stop
                if msg["message_type"] in ["auth_error", "quota_exceeded_error"]:
                    error_msg = f"ElevenLabs API Error: {msg.get('error')}"
                    raise RuntimeError(error_msg)

            await self._handle_message(msg, receiver, turn_counter, session_started)

            # Increment turn counter if we just finished a turn
            msg_type = msg.get("message_type")
            if msg_type in [
                "committed_transcript",
                "committed_transcript_with_timestamps",
            ]:
                turn_counter += 1

    async def _handle_message(
        self,
        msg: Any,
        receiver: TranscriptReceiver,
        turn_id: int,
        session_started: asyncio.Event,
    ) -> None:
        match msg:
            case {"message_type": "session_started"}:
                logger.info("ElevenLabs Session started: %s", msg.get("session_id"))
                session_started.set()  # Signal that we can start sending audio

            case {"message_type": "partial_transcript", "text": text}:
                # Interim result
                if text:
                    turn = Turn(
                        id=turn_id,
                        text=text,
                        final=False,
                    )
                    await receiver.receive_turn(turn)

            case {
                "message_type": "committed_transcript_with_timestamps",
                "text": text,
                "words": words_data,
            }:
                # Final result with word-level details
                words_list = []
                if words_data:
                    for w in words_data:
                        # EL timestamps are in seconds (float)
                        start_sec = w.get("start", 0.0)
                        end_sec = w.get("end", 0.0)

                        words_list.append(
                            Word(
                                type=w.get("type", "word"),  # "word" or "spacing"
                                text=w.get("text", ""),
                                start=timedelta(seconds=start_sec),
                                end=timedelta(seconds=end_sec),
                                speaker=w.get("speaker_id", ""),
                            )
                        )

                turn = Turn(
                    id=turn_id,
                    text=text,
                    final=True,
                    words=words_list,
                )
                await receiver.receive_turn(turn)

            case {"message_type": "committed_transcript", "text": text}:
                # Fallback if timestamps are disabled
                turn = Turn(
                    id=turn_id,
                    text=text,
                    final=True,
                )
                await receiver.receive_turn(turn)

            case {"error": error}:
                raise TranscribeError(error)

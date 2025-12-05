"""
Our helpers to work with the AssemblyAI API
"""

import abc
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import httpx
import websockets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    """
    Deep dive into a word inside a turn
    """

    text: str
    start: timedelta
    end: timedelta
    confidence: float
    word_is_final: bool


@dataclass(frozen=True)
class Turn:
    """
    Describes a turn and all the relative information
    """

    order: int
    is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: list[Word]
    utterance: str = ""
    language_code: str = ""
    language_confidence: float | None = None


class StreamReceiver(abc.ABC):
    """
    The stream receiving interface to implement if you want to receive live
    transcriptions.
    """

    @abc.abstractmethod
    async def session_begins(
        self,
        session_id: uuid.UUID,
        expires_at: datetime,
    ) -> None:
        """
        This will get called when the session begins.
        """

        raise NotImplementedError

    @abc.abstractmethod
    async def turn(self, turn: Turn) -> None:
        """
        Whenever a turn appears or gets updated, this receives a message
        """

        raise NotImplementedError

    @abc.abstractmethod
    async def termination(
        self,
        audio_duration: timedelta,
        session_duration: timedelta,
    ) -> None:
        """
        This gets called when the session ends.
        """

        raise NotImplementedError


@dataclass
class AAI:
    """
    Our own wrapper around the Assembly AI API.

    Why make it different from the official one? Because the official one
    essentially sucks:

    - It's not async, which is a massive pain in the ass
    - Type annotations are not great, especially for streaming
    - It requires you to set the API key globally somehow?
    """

    api_key: str
    region: Literal["eu", "us"] = "eu"

    @property
    def domain(self) -> str:
        """
        Domain name for the regular API
        """

        if self.region == "eu":
            return "api.eu.assemblyai.com"
        else:
            return "api.assemblyai.com"

    @property
    def streaming_domain(self) -> str:
        """
        Domain name for the streaming API
        """

        if self.region == "eu":
            return "streaming.eu.assemblyai.com"
        else:
            return "streaming.assemblyai.com"

    @asynccontextmanager
    async def client(self, streaming: bool = False):
        """
        Generates an HTTP client to communicate with the AssemblyAI

        Parameters
        ----------
        streaming: bool
            Indicates to use the streaming endpoint instead of the regular API
        """

        if streaming:
            domain = self.streaming_domain
        else:
            domain = self.domain

        async with httpx.AsyncClient(
            base_url=f"https://{domain}/",
            timeout=30,
            headers={
                "Authorization": self.api_key,
            },
        ) as client:
            yield client

    async def get_stream_token(self) -> str:
        """
        The stream API websocket requires a temporary token, that's how you get
        it. Let's note that you don't need to call this manually, stream() will
        take care of that itself.
        """

        async with self.client(streaming=True) as client:
            resp = await client.get("/v3/token", params=dict(expires_in_seconds=60))
            resp.raise_for_status()
            return resp.json()["token"]

    async def stream(
        self,
        sender: asyncio.Queue[bytes],
        receiver: StreamReceiver,
        /,
        sample_rate: int,
        encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le",
        end_of_turn_confidence_threshold: float = 0.4,
        format_turns: bool = True,
        inactivity_timeout: timedelta | None = None,
        keyterms_prompt: list[str] | None = None,
        language_detection: bool = True,
        min_end_of_turn_silence_when_confident: timedelta = timedelta(milliseconds=400),
        max_turn_silence: timedelta = timedelta(milliseconds=1280),
        speech_model: str = "universal-streaming-multilingual",
    ):
        """This calls the streaming transcription API

        The two core things are:

        - `sender` is an async queue that will get fed all the chunks of audio
          as they appear
        - `receiver` is an implementation of the StreamReceiver interface which
          will be invoked at various events, including and especially when new
          sentences (well, turns) have been heard

        All the parameters here match the parameters from the API (see:
        https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api),
        except for the fact that some defaults have been changed because they
        were too anglo-centric.
        """

        token = await self.get_stream_token()

        ws_url = httpx.URL(f"wss://{self.streaming_domain}/v3/ws").copy_merge_params(
            dict(
                sample_rate=sample_rate,
                encoding=encoding,
                end_of_turn_confidence_threshold=end_of_turn_confidence_threshold,
                format_turns="true" if format_turns else "false",
                inactivity_timeout=(
                    str(inactivity_timeout.total_seconds())
                    if inactivity_timeout
                    else None
                ),
                keyterms_prompt=keyterms_prompt or [],
                language_detection="true" if language_detection else "false",
                min_end_of_turn_silence_when_confident=str(
                    min_end_of_turn_silence_when_confident.total_seconds() * 1000
                ),
                max_turn_silence=str(max_turn_silence.total_seconds() * 1000),
                speech_model=speech_model,
                token=token,
            )
        )

        termination_received = asyncio.Event()
        should_send_terminate = asyncio.Event()

        async with websockets.connect(str(ws_url)) as ws:
            tx_t = asyncio.create_task(_stream_tx(sender, should_send_terminate, ws))
            rx_t = asyncio.create_task(_stream_rx(ws, receiver, termination_received))

            try:
                done, _ = await asyncio.wait(
                    [tx_t, rx_t],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Check if any task raised an exception
                for task in done:
                    if not task.cancelled():
                        task.result()  # This will re-raise any exception

            except asyncio.CancelledError:
                should_send_terminate.set()
                raise

            finally:
                if should_send_terminate.is_set() and not termination_received.is_set():
                    try:
                        async with asyncio.timeout(5):
                            await ws.send(json.dumps({"type": "Terminate"}))
                            await termination_received.wait()
                    except TimeoutError:
                        logger.warning("Timeout waiting for Termination response")
                    except Exception:
                        logger.exception("Failed to complete graceful termination")

                tx_t.done() or tx_t.cancel()
                rx_t.done() or rx_t.cancel()
                await asyncio.gather(tx_t, rx_t, return_exceptions=True)


async def _stream_tx(
    sender: asyncio.Queue[bytes],
    should_send_terminate: asyncio.Event,
    ws: websockets.ClientConnection,
) -> None:
    try:
        while True:
            blob = await sender.get()

            if not blob:
                should_send_terminate.set()
                return

            await ws.send(blob)
    except asyncio.CancelledError:
        should_send_terminate.set()
        raise


async def _stream_rx(
    ws: websockets.ClientConnection,
    receiver: StreamReceiver,
    term_event: asyncio.Event,
) -> None:
    while not term_event.is_set():
        msg = await ws.recv()

        try:
            msg = json.loads(msg)
        except json.decoder.JSONDecodeError:
            continue

        try:
            await _match_msg(msg, receiver, term_event)
        except Exception:
            logger.exception("Failed to process message")


async def _match_msg(
    msg: Any, receiver: StreamReceiver, term_event: asyncio.Event
) -> None:
    match msg:
        case {
            "type": "Begin",
            "id": session_id,
            "expires_at": expires_at,
        }:
            await receiver.session_begins(
                session_id=uuid.UUID(session_id),
                expires_at=datetime.fromtimestamp(expires_at).astimezone(),
            )
        case {
            "type": "Termination",
            "audio_duration_seconds": audio_duration,
            "session_duration_seconds": session_duration,
        }:
            await receiver.termination(
                audio_duration=timedelta(seconds=audio_duration),
                session_duration=timedelta(seconds=session_duration),
            )
            term_event.set()
        case {
            "type": "Turn",
            "turn_order": order,
            "turn_is_formatted": is_formatted,
            "end_of_turn": end_of_turn,
            "transcript": transcript,
            "end_of_turn_confidence": end_of_turn_confidence,
            "words": words,
        }:
            words_dec = [
                Word(
                    text=w["text"],
                    start=timedelta(milliseconds=w["start"]),
                    end=timedelta(milliseconds=w["end"]),
                    confidence=w["confidence"],
                    word_is_final=w["word_is_final"],
                )
                for w in words
            ]

            await receiver.turn(
                Turn(
                    order=order,
                    is_formatted=is_formatted,
                    end_of_turn=end_of_turn,
                    transcript=transcript,
                    end_of_turn_confidence=end_of_turn_confidence,
                    words=words_dec,
                    utterance=msg.get("utterance") or "",
                    language_code=msg.get("language_code") or "",
                    language_confidence=msg.get("language_confidence"),
                )
            )

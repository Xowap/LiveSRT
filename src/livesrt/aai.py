import abc
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import httpx
import websockets

logger = logging.getLogger(__name__)


@dataclass
class Word:
    text: str
    start: timedelta
    end: timedelta
    confidence: float
    word_is_final: bool


@dataclass(frozen=True)
class Turn:
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
    @abc.abstractmethod
    async def session_begins(
        self,
        session_id: uuid.UUID,
        expires_at: datetime,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def turn(self, turn: Turn) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def termination(
        self,
        audio_duration: timedelta,
        session_duration: timedelta,
    ) -> None:
        raise NotImplementedError


@dataclass
class AAI:
    api_key: str
    region: Literal["eu", "us"] = "eu"

    @property
    def domain(self) -> str:
        if self.region == "eu":
            return "api.eu.assemblyai.com"
        else:
            return "api.assemblyai.com"

    @property
    def streaming_domain(self) -> str:
        if self.region == "eu":
            return "streaming.eu.assemblyai.com"
        else:
            return "streaming.assemblyai.com"

    @asynccontextmanager
    async def client(self, streaming: bool = False):
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

        async def tx():
            while blob := await sender.get():
                await ws.send(blob)

        async def rx():
            while msg := await ws.recv():
                try:
                    msg = json.loads(msg)
                except json.decoder.JSONDecodeError:
                    pass

                try:
                    match msg:
                        case {
                            "type": "Begin",
                            "id": session_id,
                            "expires_at": expires_at,
                        }:
                            await receiver.session_begins(
                                session_id=uuid.UUID(session_id),
                                expires_at=datetime.fromtimestamp(expires_at),
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
                except Exception:
                    logger.exception("Failed to process message")

        async with websockets.connect(str(ws_url)) as ws:
            tx_t = asyncio.create_task(tx())
            rx_t = asyncio.create_task(rx())

            try:
                await asyncio.gather(tx_t, rx_t)
            finally:
                if not tx_t.done():
                    tx_t.cancel()
                if not rx_t.done():
                    rx_t.cancel()
                await asyncio.gather(tx_t, rx_t, return_exceptions=True)

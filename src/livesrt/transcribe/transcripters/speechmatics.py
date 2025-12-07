"""
Speechmatics implementation of the Transcripter interface.
"""

import asyncio
import json
import logging
from dataclasses import field
from datetime import timedelta
from itertools import groupby
from typing import Literal, cast

import websockets
from pydantic.dataclasses import dataclass

from ...errors import TranscribeError
from ..base import AudioSource, Transcripter, TranscriptReceiver, Turn, Word

logger = logging.getLogger(__name__)

# Bidi control characters
LRI = "\u2066"  # Left-to-Right Isolate
RLI = "\u2067"  # Right-to-Left Isolate
PDI = "\u2069"  # Pop Directional Isolate


@dataclass
class WordDisplay:
    """
    Helps to know the display direction of said word
    """

    direction: Literal["ltr", "rtl"]


@dataclass
class WordAlternative:
    """
    One alternative for a Speechmatic word
    """

    content: str
    confidence: float
    language: str
    speaker: str
    display: WordDisplay = field(default_factory=lambda: WordDisplay("ltr"))
    tags: list[str] = field(default_factory=list)


@dataclass
class TempWord:
    """
    Word representation from Speechmatics
    """

    type: Literal["word", "punctuation", "entity"]
    start_time: float
    end_time: float
    alternatives: list[WordAlternative]
    attaches_to: Literal["next", "previous", "both", "none"] = "none"
    is_eos: bool = False
    entity_class: str = ""


def join_words(
    words: list[TempWord],
    natural_direction: Literal["rtl", "ltr"] = "ltr",
) -> str:
    """
    Joins the words from the transcript following the Speechmatics semantics.
    """

    out = ""

    if not words:
        return out

    for i, (direction, segment) in enumerate(
        groupby(
            words,
            key=lambda w: w.alternatives[0].display.direction,
        )
    ):
        text = ""

        for j, word in enumerate(segment):
            if word.attaches_to in ["next", "none"] and j > 0:
                text += " "

            text += word.alternatives[0].content

        if i > 0:
            out += " "

        if i > 0 or direction != natural_direction:
            symbol = RLI if direction == "rtl" else LRI
            out += f"{symbol}{text}{PDI}"
        else:
            out += text

    return out.strip()


def transform_as_words(words: list[TempWord]) -> list[Word]:
    """
    Transforms the Speechmatics words into our (simpler) internal format
    """

    out: list[Word] = []

    if not words:
        return out

    for i, word in enumerate(words):
        if word.attaches_to in ["next", "none"] and i > 0:
            out.append(
                Word(
                    type="spacing",
                    text=word.alternatives[0].content,
                    speaker=word.alternatives[0].speaker,
                    start=timedelta(seconds=word.start_time),
                    end=timedelta(seconds=word.start_time),
                )
            )

        out.append(
            Word(
                type=cast(
                    "Literal['word', 'spacing', 'punctuation']",
                    {
                        "word": "word",
                        "punctuation": "punctuation",
                        "entity": "word",
                    }[word.type],
                ),
                text=word.alternatives[0].content,
                speaker=word.alternatives[0].speaker,
                start=timedelta(seconds=word.start_time),
                end=timedelta(seconds=word.start_time),
            )
        )

    return out


@dataclass
class TranscriptBuilder:
    """
    Helps post-processing transcript messages from Speechmatics
    """

    partial: list[TempWord] = field(default_factory=list)
    total: list[TempWord] = field(default_factory=list)

    def add_words(self, type_: Literal["partial", "total"], words: list[TempWord]):
        """
        Adds words to the transcript, potentially replacing words that were
        already in there.

        Parameters
        ----------
        type_
            Group of words to replace
        words
            The words themselves
        """

        if not words:
            return

        min_t = min(w.start_time for w in words)
        max_t = max(w.end_time for w in words)

        keep = [
            w
            for w in getattr(self, type_)
            if w.end_time <= min_t or max_t <= w.start_time
        ]

        setattr(self, type_, sorted([*keep, *words], key=lambda w: w.start_time))

    @property
    def combined(self) -> list[TempWord]:
        """
        Combines partial and total transcripts into one coherent transcript.
        """

        if not self.total:
            return self.partial

        cutoff = self.total[-1].end_time

        return [
            *self.total,
            *[w for w in self.partial if w.start_time >= cutoff],
        ]

    def generate(self):
        """
        Generates the plain text for this utterance
        """

        return join_words(self.combined)

    def clear(self):
        """
        Reset the utterance and start a new one
        """

        self.partial = []
        self.total = []


@dataclass
class SpeechmaticsTranscripter(Transcripter):
    """
    Transcribes audio using the Speechmatics Realtime API.
    """

    api_key: str
    language: str
    url: str = "wss://eu2.rt.speechmatics.com/v2"
    tb: TranscriptBuilder = field(default_factory=TranscriptBuilder)
    turn_id: int = field(default=0, init=False)

    async def process(self, source: AudioSource, receiver: TranscriptReceiver) -> None:
        """
        Main processing loop.
        """
        # Speechmatics uses Authorization header
        headers = {"Authorization": f"Bearer {self.api_key}"}

        termination_received = asyncio.Event()
        session_started = asyncio.Event()
        should_send_terminate = asyncio.Event()

        # Notify receiver we are starting
        await receiver.start()

        async with websockets.connect(self.url, additional_headers=headers) as ws:
            tx_t = asyncio.create_task(
                self._stream_tx(source, ws, session_started, should_send_terminate)
            )
            rx_t = asyncio.create_task(
                self._stream_rx(ws, receiver, termination_received, session_started)
            )

            try:
                done, _ = await asyncio.wait(
                    [tx_t, rx_t],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in done:
                    if not task.cancelled():
                        task.result()

            except asyncio.CancelledError:
                should_send_terminate.set()
                raise

            finally:
                should_send_terminate.set()

                # Ensure tasks are cleaned up
                tx_t.done() or tx_t.cancel()
                rx_t.done() or rx_t.cancel()
                await asyncio.gather(tx_t, rx_t, return_exceptions=True)

                await receiver.stop()

    async def _stream_tx(
        self,
        source: AudioSource,
        ws: websockets.ClientConnection,
        session_started: asyncio.Event,
        should_send_terminate: asyncio.Event,
    ) -> None:
        """
        Transmission loop: Sends StartRecognition, waits for ack, then streams audio.
        """
        # 1. Send configuration message
        sample_rate = source.get_sample_rate()
        encoding = "pcm_s16le" if source.get_format() == "pcm" else "mulaw"

        start_message = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": encoding,
                "sample_rate": sample_rate,
            },
            "transcription_config": {
                "language": self.language,
                "enable_partials": True,
                "diarization": "speaker",
                "operating_point": "enhanced",
                "max_delay_mode": "flexible",
                "conversation_config": {"end_of_utterance_silence_trigger": 0.5},
            },
        }

        await ws.send(json.dumps(start_message))

        # 2. Wait for RecognitionStarted from RX loop
        await session_started.wait()

        # 3. Stream Audio
        last_seq_no = 0
        try:
            async for chunk in source.iter_frames():
                if should_send_terminate.is_set():
                    break

                if not chunk:
                    break

                # Send binary data directly
                await ws.send(chunk)
                last_seq_no += 1

            # 4. End of Stream
            eos_message = {"message": "EndOfStream", "last_seq_no": last_seq_no}
            await ws.send(json.dumps(eos_message))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in Speechmatics transmission loop")
            raise

    async def _stream_rx(
        self,
        ws: websockets.ClientConnection,
        receiver: TranscriptReceiver,
        term_event: asyncio.Event,
        session_started: asyncio.Event,
    ) -> None:
        """
        Reception loop: Reads JSON messages.
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

            await self._handle_message(msg, receiver, session_started, term_event)

    async def _handle_message(
        self,
        msg,
        receiver: TranscriptReceiver,
        session_started: asyncio.Event,
        term_event: asyncio.Event,
    ) -> None:
        match msg:
            case {"message": "AddPartialTranscript", "results": results}:
                self.tb.add_words("partial", [TempWord(**w) for w in results])
                await self._update_turn(receiver)
            case {"message": "AddTranscript", "results": results}:
                self.tb.add_words("total", [TempWord(**w) for w in results])
                await self._update_turn(receiver)
            case {"message": "EndOfUtterance"}:
                await self._finish_turn(receiver)
            case {"message": "RecognitionStarted"}:
                session_started.set()
            case {"message": "EndOfTranscript"}:
                term_event.set()
            case {"message": "Error", "type": error_type}:
                err = f"Speechmatics error: {error_type}"
                raise TranscribeError(err)

    async def _update_turn(
        self, receiver: TranscriptReceiver, final: bool = False
    ) -> None:
        await receiver.receive_turn(
            Turn(
                id=self.turn_id,
                text=self.tb.generate(),
                final=final,
                words=transform_as_words(self.tb.combined),
            )
        )

    async def _finish_turn(self, receiver: TranscriptReceiver) -> None:
        await self._update_turn(receiver, final=True)
        self.turn_id += 1
        self.tb.clear()

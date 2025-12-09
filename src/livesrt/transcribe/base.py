"""
Here we define the base interfaces to implement in order to make a transcription
service.
"""

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

AudioFormat = Literal["pcm", "ulaw"]
AudioSample = Literal[8000, 16000, 22050, 24000, 44100, 48000]
AudioDepth = Literal[16]


class AudioSource(abc.ABC):
    """
    Implement this to provide a source of audio for the purpose of transcription
    """

    @abc.abstractmethod
    def get_format(self) -> AudioFormat:
        """
        Indicates the format of the stream
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_sample_rate(self) -> AudioSample:
        """
        Returns the sample rate of the audio
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_depth(self) -> AudioDepth:
        """
        Returns the bit depth of the audio stream
        """

        raise NotImplementedError

    @abc.abstractmethod
    def iter_frames(self) -> AsyncIterator[bytes]:
        """
        Iterates over audio frames. It's up to the source to size them
        reasonably. Overall, the audio source can do a lot of shit, and it's
        up to the implementer to figure settings that are going to fly with the
        various transcripter services. There is no particular safeguards.
        """

        raise NotImplementedError


@dataclass
class Word:
    """
    One uttered word. Depending on the service, you can have more or less
    information about it.

    Parameters
    ----------
    type
        Could be a word or a spacing/silence between words
    text
        The text representing said word
    start
        Duration between the start of the session and the start of this word
    end
        Duration between the start of the session and the end of this word
    speaker
        A speaker label (if identified)
    """

    type: Literal["word", "spacing", "punctuation"]
    text: str
    start: timedelta | None = None
    end: timedelta | None = None
    speaker: str = ""


@dataclass
class Turn:
    """
    Represents an audio turn. That's a loose definition, essentially it just
    means that it's a relatively semantic chunk of conversation, uttered by one
    or several people and made of one or several sentences. Well, speech is not
    exactly bound to typographical conventions, is it?
    """

    id: int
    text: str
    final: bool
    words: list[Word] = field(default_factory=list)
    debug: list[dict] = field(default_factory=list)


class TranscriptReceiver(abc.ABC):
    """
    Implement this interface to receive live transcriptions from any
    implementation of the Transcripter class
    """

    async def start(self) -> None:
        """
        Indicates that the transcription started
        """
        return

    async def stop(self) -> None:
        """
        Indicates that the transcription stopped
        """
        return

    @abc.abstractmethod
    async def receive_turn(self, turn: Turn) -> None:
        """
        Receive here a turn from the transcripter service. The turn can either
        be final or temporary. If temporary, you'll probably receive the final
        one shortly after (with the same ID).
        """

        raise NotImplementedError


class Transcripter(abc.ABC):
    """
    Implement this interface so that you can transcribe audio into text
    """

    @abc.abstractmethod
    async def process(self, source: AudioSource, receiver: TranscriptReceiver) -> None:
        """
        This function runs the source through the transcription service and
        notifies the receiver when something is decoded. It runs forever, until
        the source runs out or until the service cuts the connection, whichever
        happens first.
        """
        raise NotImplementedError

"""This is the module in charge of live audio capture"""

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta

import pyaudio

from livesrt.utils import ignore_stderr

from ..base import AudioDepth, AudioFormat, AudioSample, AudioSource


@dataclass(frozen=True)
class MicInfo:
    """
    Meta-information regarding a microphone
    """

    index: int
    name: str


def make_pyaudio():
    """Custom init for PyAudio so that it can stfu"""

    with ignore_stderr():
        return pyaudio.PyAudio()


@dataclass(frozen=True)
class StreamConfig:
    """
    Configuration for microphone streaming.

    Parameters
    ----------
    frames_per_buffer : int
        Number of frames per read. Controls the granularity/latency of audio chunks.
    queue_size : int
        Maximum buffered chunks. When the consumer falls behind by this many chunks,
        older data will be dropped.
    """

    frames_per_buffer: int
    queue_size: int

    @classmethod
    def from_time_params(
        cls,
        sample_rate: int,
        buffer_duration: timedelta = timedelta(milliseconds=100),
        max_latency: timedelta = timedelta(seconds=3),
    ) -> "StreamConfig":
        """
        Create a StreamConfig from time-based parameters.
        """

        buffer_seconds = buffer_duration.total_seconds()
        latency_seconds = max_latency.total_seconds()

        frames_per_buffer = int(sample_rate * buffer_seconds)
        queue_size = int(latency_seconds / buffer_seconds)

        return cls(frames_per_buffer=frames_per_buffer, queue_size=queue_size)


@dataclass
class MicSource(AudioSource):
    """
    Audio source that captures from a microphone device.
    """

    device_index: int | None
    sample_rate: AudioSample
    config: StreamConfig
    p: pyaudio.PyAudio
    _run: bool = field(default=False, init=False, repr=False)
    _queue: asyncio.Queue[bytes] | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def get_format(self) -> AudioFormat:
        """Keepin' our frequencies"""
        return "pcm"

    def get_sample_rate(self) -> AudioSample:
        """The user's the boss"""
        return self.sample_rate

    def get_depth(self) -> AudioDepth:
        """No choice there"""
        return 16

    async def iter_frames(self) -> AsyncIterator[bytes]:
        """
        Stream audio frames from the microphone.

        Yields
        ------
        bytes
            Raw audio data chunks.
        """

        self._run = True
        self._queue = asyncio.Queue(self.config.queue_size)
        loop = asyncio.get_event_loop()

        def _stream() -> None:
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.config.frames_per_buffer,
            )

            try:
                while self._run and (
                    data := stream.read(
                        self.config.frames_per_buffer,
                        exception_on_overflow=False,
                    )
                ):
                    if self._queue:
                        asyncio.run_coroutine_threadsafe(
                            self._queue.put(data), loop=loop
                        ).result()
            finally:
                stream.stop_stream()
                stream.close()

        self._thread = threading.Thread(target=_stream, daemon=True)
        self._thread.start()

        try:
            while True:
                data = await self._queue.get()
                if not data:
                    break
                yield data
        finally:
            self._run = False
            if self._thread:
                self._thread.join(timeout=1.0)

    async def health_check(self) -> None:
        """Checks if the microphone device is available."""

        # This runs in a thread to avoid blocking the loop with PyAudio
        def _check():
            try:
                if self.device_index is not None:
                    self.p.get_device_info_by_index(self.device_index)
                else:
                    self.p.get_default_input_device_info()
            except OSError as e:
                # PyAudio raises OSError if device not found
                error_msg = f"Microphone device error: {e}"
                raise ValueError(error_msg) from e

        await asyncio.to_thread(_check)

    @property
    def name(self) -> str:
        """Returns a friendly name for the audio source."""
        try:
            if self.device_index is not None:
                info = self.p.get_device_info_by_index(self.device_index)
                return f"Mic #{self.device_index} ({info.get('name')})"
            else:
                info = self.p.get_default_input_device_info()
                return f"Default Mic ({info.get('name')})"
        except Exception:
            return f"Mic #{self.device_index}" if self.device_index else "Default Mic"

    def get_settings(self) -> dict[str, str]:
        """Returns a dictionary of relevant settings for display."""
        return {
            **super().get_settings(),
            "Buffer Duration": str(self.config.frames_per_buffer / self.sample_rate)
            + "s",
        }


@dataclass
class MicSourceFactory:
    """
    Factory for creating microphone audio sources.

    Parameters
    ----------
    sample_rate : Literal[8000, 16000, 22050, 24000, 44100, 48000]
        Sample rate in Hz for audio capture (default: 16000)
    segment_duration : timedelta
        Duration of each audio segment/chunk (default: 100ms)
    max_lag_duration : timedelta
        Maximum allowed lag before dropping old data (default: 3s)
    """

    sample_rate: AudioSample = 16_000
    segment_duration: timedelta = field(
        default_factory=lambda: timedelta(milliseconds=100)
    )
    max_lag_duration: timedelta = field(default_factory=lambda: timedelta(seconds=3))
    p: pyaudio.PyAudio = field(default_factory=make_pyaudio, repr=False)

    def list_devices(self) -> dict[int, MicInfo]:
        """
        Lists all available input devices.

        Returns
        -------
        dict[int, MicInfo]
            Dictionary mapping device index to device information.
        """

        out: dict[int, MicInfo] = {}

        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_index(i)

            if int(device["maxInputChannels"]) > 0:
                out[i] = MicInfo(
                    index=i,
                    name=str(device["name"]),
                )

        return out

    def is_device_valid(self, index: int) -> bool:
        """
        Checks if the device index corresponds to a valid input device.

        Parameters
        ----------
        index : int
            Device index to check.

        Returns
        -------
        bool
            True if the device is valid, False otherwise.
        """

        devices = self.list_devices()
        return index in devices

    def create_source(self, device_index: int | None = None) -> MicSource:
        """
        Create a microphone audio source.

        Parameters
        ----------
        device_index : int or None
            Device index to use, or None for the default device.

        Returns
        -------
        MicSource
            Configured microphone audio source.
        """

        config = StreamConfig.from_time_params(
            sample_rate=self.sample_rate,
            buffer_duration=self.segment_duration,
            max_latency=self.max_lag_duration,
        )

        return MicSource(
            device_index=device_index,
            sample_rate=self.sample_rate,
            config=config,
            p=self.p,
        )

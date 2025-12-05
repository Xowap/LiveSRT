"""This is the module in charge of live audio capture"""

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import pyaudio

from .async_tools import sync_to_async


@dataclass
class MicInfo:
    """
    Meta-information regarding a microphone
    """

    index: int
    name: str


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

    Notes
    -----
    Prefer using `MicManager.make_stream_config` if you have a manager instance,
    or `from_time_params` to specify configuration in time units rather than frames.
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

        Notes
        -----
        If you have a `MicManager` instance, use `MicManager.make_stream_config`
        instead to avoid passing the sample rate manually.
        """

        buffer_seconds = buffer_duration.total_seconds()
        latency_seconds = max_latency.total_seconds()

        frames_per_buffer = int(sample_rate * buffer_seconds)
        queue_size = int(latency_seconds / buffer_seconds)

        return cls(frames_per_buffer=frames_per_buffer, queue_size=queue_size)


@dataclass
class MicManager:
    """
    Wrapper around PyAudio for convenient microphone access.
    """

    sample_rate: int = 16_000
    p: pyaudio.PyAudio = field(init=False, default_factory=pyaudio.PyAudio)

    def make_stream_config(
        self,
        buffer_duration: timedelta = timedelta(milliseconds=100),
        max_latency: timedelta = timedelta(seconds=3),
    ) -> StreamConfig:
        """
        Create a StreamConfig using this manager's sample rate.

        Notes
        -----
        Convenience wrapper around `StreamConfig.from_time_params` that
        automatically uses the correct sample rate.
        """
        return StreamConfig.from_time_params(
            sample_rate=self.sample_rate,
            buffer_duration=buffer_duration,
            max_latency=max_latency,
        )

    @sync_to_async
    def list_microphones(self) -> dict[int, MicInfo]:
        """Lists all available input devices."""

        out: dict[int, MicInfo] = {}

        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_index(i)

            if int(device["maxInputChannels"]) > 0:
                out[i] = MicInfo(
                    index=i,
                    name=str(device["name"]),
                )

        return out

    async def is_device_valid(self, index: int) -> bool:
        """Checks if the device index corresponds to a valid input device."""

        devices = await self.list_microphones()
        return index in devices

    @asynccontextmanager
    async def stream_mic(
        self,
        index: int | None,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[asyncio.Queue[bytes]]:
        """
        Opens a microphone and streams audio data through an async queue.

        Parameters
        ----------
        index : int or None
            Device index, or None for default device.
        config : StreamConfig or None
            Stream configuration. Defaults to 100ms reads with 3s max latency.

        Notes
        -----
        The stream runs in a dedicated thread. Audio format is always 16-bit PCM
        mono at the manager's sample rate. The stream automatically stops and
        cleans up when exiting the context.
        """

        if config is None:
            config = self.make_stream_config()

        run = True
        queue: asyncio.Queue[bytes] = asyncio.Queue(config.queue_size)
        loop = asyncio.get_event_loop()

        def _stream() -> None:
            nonlocal run

            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=index,
                frames_per_buffer=config.frames_per_buffer,
            )

            while run and (
                data := stream.read(
                    config.frames_per_buffer,
                    exception_on_overflow=False,
                )
            ):
                asyncio.run_coroutine_threadsafe(queue.put(data), loop=loop).result()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        try:
            yield queue
        finally:
            run = False

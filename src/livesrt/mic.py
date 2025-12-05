"""This is the module in charge of live audio capture"""

import asyncio
import contextlib
import os
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pyaudio

from .async_tools import sync_to_async


@dataclass
class MicInfo:
    """
    Meta-information regarding a microphone
    """

    index: int
    name: str


@contextlib.contextmanager
def ignore_stderr():
    """
    Diverts stderr because by default lots of junk gets logged by alsa and
    friends for no fucking reason.
    """

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


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
    p: pyaudio.PyAudio = field(init=False, default_factory=make_pyaudio)

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

    @asynccontextmanager
    async def stream_file(
        self,
        file_path: str | Path,
        config: StreamConfig | None = None,
    ) -> AsyncIterator[asyncio.Queue[bytes]]:
        """
        Opens an audio file and streams audio data through an async queue.

        Parameters
        ----------
        file_path : str or Path
            Path to the audio file to stream.
        config : StreamConfig or None
            Stream configuration. Defaults to 100ms reads with 3s max latency.

        Notes
        -----
        Audio is converted to 16-bit PCM mono at the manager's sample rate using
        ffmpeg. The stream automatically stops and cleans up when exiting the context.
        Data is streamed at approximately real-time speed to avoid filling the queue
        faster than it would be consumed in live playback.
        """

        if config is None:
            config = self.make_stream_config()

        queue: asyncio.Queue[bytes] = asyncio.Queue(config.queue_size)

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *["-i", str(file_path)],
            *["-f", "wav"],
            *["-acodec", "pcm_s16le"],
            *["-ar", str(self.sample_rate)],
            *["-ac", "1"],
            "-hide_banner",
            *["-loglevel", "error"],
            "-",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        async def _stream() -> None:
            buf = b""
            chunk_size = 2_000
            # Calculate real-time duration for each chunk
            # 16-bit PCM = 2 bytes per sample, mono = 1 channel
            bytes_per_second = self.sample_rate * 2
            chunk_duration = chunk_size / bytes_per_second

            try:
                while process.returncode is None:
                    assert process.stdout is not None  # noqa: S101

                    while data := await process.stdout.read(chunk_size):
                        buf += data

                        while len(buf) >= chunk_size:
                            await queue.put(buf[:chunk_size])
                            buf = buf[chunk_size:]
                            await asyncio.sleep(chunk_duration)
            finally:
                await queue.put(b"")

                if process.returncode == 1:
                    if process.stderr:
                        msg = f"ffmpeg error: {process.stderr.read()}"
                    else:
                        msg = "ffmpeg error"

                    raise RuntimeError(msg)

        stream_t = asyncio.create_task(_stream())

        try:
            yield queue
        finally:
            stream_t.done() or stream_t.cancel()
            process.terminate()

            try:
                async with asyncio.timeout(5):
                    await stream_t
                    await process.wait()
            except TimeoutError:
                if process.returncode is not None:
                    process.kill()

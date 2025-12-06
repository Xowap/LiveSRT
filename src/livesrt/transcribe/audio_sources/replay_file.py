"""This is the module in charge of audio file streaming"""

import asyncio
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Literal

from ..base import AudioDepth, AudioFormat, AudioSample, AudioSource


@dataclass(frozen=True)
class StreamConfig:
    """
    Configuration for file streaming.

    Parameters
    ----------
    chunk_size : int
        Number of bytes per chunk. Controls the granularity of audio chunks.
    queue_size : int
        Maximum buffered chunks. When the consumer falls behind by this many chunks,
        older data will be dropped.
    """

    chunk_size: int
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

        # 16-bit PCM = 2 bytes per sample, mono = 1 channel
        bytes_per_second = sample_rate * 2
        chunk_size = int(bytes_per_second * buffer_seconds)
        queue_size = int(latency_seconds / buffer_seconds)

        return cls(chunk_size=chunk_size, queue_size=queue_size)


@dataclass
class FileSource(AudioSource):
    """
    Audio source that streams from an audio file.
    """

    file_path: Path
    sample_rate: AudioSample
    config: StreamConfig
    realtime: bool = True
    _process: asyncio.subprocess.Process | None = field(
        default=None, init=False, repr=False
    )
    _stream_task: asyncio.Task | None = field(default=None, init=False, repr=False)

    def get_format(self) -> AudioFormat:
        """ulaw is cool but I want all the frequencies"""
        return "pcm"

    def get_sample_rate(self) -> AudioSample:
        """Following what we're asked"""
        return self.sample_rate

    def get_depth(self) -> AudioDepth:
        """There's only one option, so..."""
        return 16

    async def _stream(self, queue: asyncio.Queue[bytes]) -> None:
        """
        Core of the streaming logic, which essentially consists of running
        ffmpeg and getting the output from it. The "complicated" part is to read
        reasonably-sized chunks and not spam them right away, given that
        transcribers do not expect to receive all at once (it's a live service)
        """

        buf = b""
        # Calculate real-time duration for each chunk
        # 16-bit PCM = 2 bytes per sample, mono = 1 channel
        bytes_per_second = self.sample_rate * 2
        chunk_duration = self.config.chunk_size / bytes_per_second

        try:
            while self._process and self._process.returncode is None:
                assert self._process.stdout is not None  # noqa: S101

                if data := await self._process.stdout.read(self.config.chunk_size):
                    buf += data

                    while len(buf) >= self.config.chunk_size:
                        await queue.put(buf[: self.config.chunk_size])
                        buf = buf[self.config.chunk_size :]
                        if self.realtime:
                            await asyncio.sleep(chunk_duration)

            # Put any remaining data
            if buf:
                await queue.put(buf)

        finally:
            await queue.put(b"")

            if self._process and self._process.returncode == 1:
                if self._process.stderr:
                    stderr_data = await self._process.stderr.read()
                    msg = f"ffmpeg error: {stderr_data.decode()}"
                else:
                    msg = "ffmpeg error"

                raise RuntimeError(msg)

    async def iter_frames(self) -> AsyncIterator[bytes]:
        """
        Stream audio frames from the file.

        Yields
        ------
        bytes
            Raw audio data chunks.
        """

        queue: asyncio.Queue[bytes] = asyncio.Queue(self.config.queue_size)

        self._process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            *["-i", str(self.file_path)],
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

        self._stream_task = asyncio.create_task(self._stream(queue))

        try:
            while True:
                data = await queue.get()
                if not data:
                    break
                yield data
        finally:
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
            if self._process:
                self._process.terminate()

            try:
                async with asyncio.timeout(5):
                    if self._stream_task:
                        await self._stream_task
                    if self._process:
                        await self._process.wait()
            except TimeoutError:
                if self._process and self._process.returncode is None:
                    self._process.kill()


@dataclass
class FileSourceFactory:
    """
    Factory for creating file audio sources.

    Parameters
    ----------
    sample_rate : Literal[8000, 16000, 22050, 24000, 44100, 48000]
        Sample rate in Hz for audio capture (default: 16000)
    segment_duration : timedelta
        Duration of each audio segment/chunk (default: 100ms)
    max_lag_duration : timedelta
        Maximum allowed lag before dropping old data (default: 3s)
    realtime : bool
        Whether to stream at real-time speed or as fast as possible (default: True)
    """

    sample_rate: Literal[8000, 16000, 22050, 24000, 44100, 48000] = 16_000
    segment_duration: timedelta = field(
        default_factory=lambda: timedelta(milliseconds=100)
    )
    max_lag_duration: timedelta = field(default_factory=lambda: timedelta(seconds=3))
    realtime: bool = True

    def create_source(self, file_path: str | Path) -> FileSource:
        """
        Create a file audio source.

        Parameters
        ----------
        file_path : str or Path
            Path to the audio file to stream.

        Returns
        -------
        FileSource
            Configured file audio source.
        """

        config = StreamConfig.from_time_params(
            sample_rate=self.sample_rate,
            buffer_duration=self.segment_duration,
            max_latency=self.max_lag_duration,
        )

        return FileSource(
            file_path=Path(file_path),
            sample_rate=self.sample_rate,
            config=config,
            realtime=self.realtime,
        )

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pyaudio

from .async_tools import sync_to_async


@dataclass
class MicInfo:
    index: int
    name: str


@dataclass
class MicManager:
    """
    Manages microphones, essentially a wrapper around PyAudio, but that makes it
    convenient to use.
    """

    sample_rate: int = 16_000
    p: pyaudio.PyAudio = field(init=False, default_factory=pyaudio.PyAudio)

    @sync_to_async
    def list_microphones(self) -> dict[int, MicInfo]:
        """
        Lists all available microphones as dictionaries
        """

        out: dict[int, MicInfo] = {}

        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_index(i)

            if device["maxInputChannels"] > 0:
                out[i] = MicInfo(
                    index=i,
                    name=device["name"],
                )

        return out

    async def is_device_valid(self, index: int) -> bool:
        """
        Checks if said device is a valid microphone (at least for our use case)
        """

        devices = await self.list_microphones()
        return index in devices

    @asynccontextmanager
    async def stream_mic(
        self, index: int | None
    ) -> AsyncIterator[asyncio.Queue[bytes]]:
        run = True
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream() -> None:
            nonlocal run

            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=index,
                frames_per_buffer=8192,
            )

            while run and (data := stream.read(8192, exception_on_overflow=False)):
                asyncio.run_coroutine_threadsafe(queue.put(data), loop=loop).result()

        t = threading.Thread(target=_stream, daemon=True)
        t.start()

        try:
            yield queue
        finally:
            run = False

from abc import ABC, abstractmethod
from typing import AsyncGenerator
from multiprocessing.synchronize import Event
from PIL import Image

class StreamProtocol(ABC):
    @abstractmethod
    async def start(self):
        """Initialize and start the streaming protocol"""
        pass

    @abstractmethod
    async def stop(self):
        """Clean up and stop the streaming protocol"""
        pass

    @abstractmethod
    async def ingress_loop(self, done: Event) -> AsyncGenerator[Image.Image, None]:
        """Generator that yields the ingress frames"""
        if False:
            yield Image.new('RGB', (1, 1))  # dummy yield for type checking
        pass

    @abstractmethod
    async def egress_loop(self, output_frames: AsyncGenerator[Image.Image, None]):
        """Consumes generated frames and processes them"""
        pass

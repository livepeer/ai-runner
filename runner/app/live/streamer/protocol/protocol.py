import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from PIL import Image

from ...trickle import InputFrame, OutputFrame

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
    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        """Generator that yields the ingress frames"""
        if False:
            yield Image.new('RGB', (1, 1))  # dummy yield for type checking
        pass

    @abstractmethod
    async def egress_loop(self, output_frames: AsyncGenerator[OutputFrame, None]):
        """Consumes generated frames and processes them"""
        pass

    @abstractmethod
    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Sends a monitoring event to the event stream if available"""
        pass

    @abstractmethod
    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        """Generator that yields control messages for updating pipeline parameters"""
        if False:
            yield {}  # dummy yield for proper typing
        pass

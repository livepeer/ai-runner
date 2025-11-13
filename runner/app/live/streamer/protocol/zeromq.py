import io
import time
import asyncio
from typing import AsyncGenerator
from fractions import Fraction

import zmq.asyncio
from PIL import Image
import numpy as np
import torch

from ...trickle import InputFrame, VideoOutput
from .protocol import StreamProtocol

TIME_BASE = Fraction(1, 90000)

class ZeroMQProtocol(StreamProtocol):
    def __init__(self, input_address: str, output_address: str):
        self.input_address = input_address
        self.output_address = output_address
        self.context = zmq.asyncio.Context()
        self.input_socket = self.context.socket(zmq.SUB)
        self.output_socket = self.context.socket(zmq.PUB)

    async def start(self):
        self.input_socket.connect(self.input_address)
        self.input_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.input_socket.set_hwm(10)
        self.output_socket.connect(self.output_address)
        self.output_socket.set_hwm(10)

    async def stop(self):
        self.input_socket.close()
        self.output_socket.close()
        self.context.term()

    async def ingress_loop(self, done: asyncio.Event) -> AsyncGenerator[InputFrame, None]:
        while not done.is_set():
            frame_bytes = await self.input_socket.recv()
            tensor = from_jpeg_bytes(frame_bytes)
            ts = int(time.time() / TIME_BASE)
            frame = InputFrame.from_av_video(tensor, ts, TIME_BASE)
            yield frame

    async def egress_loop(self, output_frames: AsyncGenerator[VideoOutput, None]):
        async for output in output_frames:
            frame_bytes = to_jpeg_bytes(output.tensor)
            await self.output_socket.send(frame_bytes)

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        pass  # No-op for ZeroMQ

    async def control_loop(self, done: asyncio.Event) -> AsyncGenerator[dict, None]:
        if False:
            yield {}  # Empty generator, dummy yield for proper typing
        await done.wait() # ZeroMQ protocol does not support control messages so just wait for the stop event


def to_jpeg_bytes(tensor: torch.Tensor):
    image_np = (tensor * 255).byte().cpu().numpy()
    image = Image.fromarray(image_np)

    buffer = io.BytesIO()
    try:
        image.save(buffer, format="JPEG")
        return buffer.getvalue()
    finally:
        buffer.close()


def from_jpeg_bytes(frame_bytes: bytes):
    image = Image.open(io.BytesIO(frame_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if (width, height) != (512, 512):
        # Crop to the center square if image not already square
        square_size = 512
        start_x = width // 2 - square_size // 2
        start_y = height // 2 - square_size // 2
        image = image.crop((start_x, start_y, start_x + square_size, start_y + square_size))

    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.tensor(image_np).unsqueeze(0)

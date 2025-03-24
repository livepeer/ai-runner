import logging
import asyncio
from PIL import Image


from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)
    self.frame_queue = asyncio.Queue()

  async def put_video_frame(self, frame: VideoFrame):
    await self.frame_queue.put(frame)

  async def get_video_frame(self) -> VideoOutput:
    frame = await self.frame_queue.get()
    processed_frame = frame.image.convert("RGB")
    return VideoOutput(frame.replace_image(processed_frame))

  def update_params(self, **params):
    logging.info(f"Updating params: {params}")

  async def stop(self):
    logging.info("Stopping pipeline")

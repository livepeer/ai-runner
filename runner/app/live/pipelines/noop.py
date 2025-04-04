import logging
import asyncio
from PIL import Image


from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self):
    self.frame_queue = asyncio.Queue()

  async def put_video_frame(self, frame: VideoFrame):
    await self.frame_queue.put(frame)

  async def get_processed_video_frame(self) -> VideoFrame:
    frame = await self.frame_queue.get()
    processed_frame = frame.image.convert("RGB")
    return frame.replace_image(processed_frame)

  async def warm_video(self):
    logging.info("Warming video")

  async def set_params(self, **params):
    logging.info(f"Setting params: {params}")

  async def update_params(self, **params):
    logging.info(f"Updating params: {params}")

  async def stop(self):
    logging.info("Stopping pipeline")

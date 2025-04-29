import logging
import asyncio
from PIL import Image


from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self):
    self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()

  async def put_video_frame(self, frame: VideoFrame, request_id: str):
    await self.frame_queue.put(VideoOutput(frame, request_id))

  async def get_processed_video_frame(self, request_id: str) -> VideoOutput:
    out = await self.frame_queue.get()
    processed_frame = out.image.convert("RGB")
    return VideoOutput(out.frame.replace_image(processed_frame), request_id)

  async def initialize(self, **params):
    logging.info(f"Initializing Noop pipeline with params: {params}")
    logging.info("Pipeline initialization complete")

  async def update_params(self, **params):
    logging.info(f"Updating params: {params}")

  async def stop(self):
    logging.info("Stopping pipeline")

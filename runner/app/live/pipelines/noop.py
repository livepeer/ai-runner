import logging
import asyncio
import time
import os

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self):
    self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
    self.timeout_started = False
    self.start_time = None

  async def put_video_frame(self, frame: VideoFrame, request_id: str):
    if not self.start_time:
      self.start_time = time.time()
    # stop producing frames after 10 seconds
    if time.time() - self.start_time < 10:
      await self.frame_queue.put(VideoOutput(frame, request_id))
    elif not hasattr(self, "skipping_frames"):
      self.skipping_frames = True
      logging.info("Skipping frames from now on")

    # kill the process after 60 seconds
    # if not self.timeout_started:
    #   self.timeout_started = True
    #   async def kill_after_timeout():
    #     await asyncio.sleep(60)
    #     os._exit(1)
    #   asyncio.create_task(kill_after_timeout())

  async def get_processed_video_frame(self) -> VideoOutput:
    out = await self.frame_queue.get()
    processed_frame = out.image.convert("RGB")
    return out.replace_image(processed_frame)

  async def initialize(self, **params):
    logging.info(f"Initializing Noop pipeline with params: {params}")
    logging.info("Pipeline initialization complete")

  async def update_params(self, **params):
    logging.info(f"Updating params: {params}")

  async def stop(self):
    logging.info("Stopping pipeline")

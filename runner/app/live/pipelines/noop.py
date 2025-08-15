import logging
import asyncio

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self):
    pass

  async def put_video_frame(self, frame: VideoFrame, request_id: str, output_cb):
    # Clone tensor for safety
    vo = VideoOutput(frame, request_id).replace_tensor(frame.tensor.clone())
    await output_cb(vo)

  async def initialize(self, **params):
    logging.info(f"Initializing Noop pipeline with params: {params}")
    logging.info("Pipeline initialization complete")

  async def update_params(self, **params):
    logging.info(f"Updating params: {params}")

  async def stop(self):
    logging.info("Stopping pipeline")

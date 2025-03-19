from PIL import Image
import logging

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

class Noop(Pipeline):
  def __init__(self, **params):
    super().__init__(**params)

  def process_frame(self, frame: VideoFrame) -> VideoOutput:
    return VideoOutput(frame.replace_image(frame.image.convert("RGB")))

  def update_params(self, **params):
    logging.info(f"Updating params: {params}")

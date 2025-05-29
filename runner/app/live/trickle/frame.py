from fractions import Fraction
import av
from PIL import Image
from typing import List
import numpy as np

class SideData:
    """
        Base class for side data, needed to keep it consistent with av frame side_data
    """
    skipped: bool = True
    input: Image.Image | np.ndarray | None

class InputFrame:
    """
    Base class for a frame (either audio or video).
    Holds any fields that may be shared across
    different frame types.
    """

    timestamp: int
    time_base: Fraction
    log_timestamps: dict[str, float] = {}
    side_data: SideData = SideData()

    @classmethod
    def from_av_video(cls, frame: av.VideoFrame):
        return VideoFrame(frame.to_image(), frame.pts, frame.time_base)

    @classmethod
    def from_av_audio(cls, frame: av.AudioFrame):
        return AudioFrame(frame)

class VideoFrame(InputFrame):
    image: Image.Image

    def __init__(self, image: Image.Image, timestamp: int, time_base: Fraction, log_timestamps: dict[str, float] = {}):
        self.image = image
        self.timestamp = timestamp
        self.time_base = time_base
        self.log_timestamps = log_timestamps

    # Returns a copy of an existing VideoFrame with its image replaced
    def replace_image(self, image: Image.Image):
        new_frame = VideoFrame(image, self.timestamp, self.time_base, self.log_timestamps)
        new_frame.side_data = self.side_data
        return new_frame

class AudioFrame(InputFrame):
    samples: np.ndarray
    format: str # av.audio.format.AudioFormat
    layout: str # av.audio.layout.AudioLayout
    rate: int
    nb_samples: int

    def __init__(self, frame: av.AudioFrame):
        if frame.pts is None:
            raise ValueError("Audio frame has no timestamp")
        self.samples = frame.to_ndarray()
        self.nb_samples = frame.samples
        self.format = frame.format.name
        self.rate = frame.sample_rate
        self.layout = frame.layout.name
        self.timestamp = frame.pts
        self.time_base = frame.time_base

class OutputFrame:
    """
        Base class for output media frames
    """
    pass

class VideoOutput(OutputFrame):
    frame: VideoFrame
    request_id: str
    def __init__(self, frame: VideoFrame, request_id: str = ''):
        self.frame = frame
        self.request_id = request_id

    def replace_image(self, image: Image.Image):
        new_frame = self.frame.replace_image(image)
        return VideoOutput(new_frame, self.request_id)

    @property
    def image(self):
        return self.frame.image

    @property
    def timestamp(self):
        return self.frame.timestamp

    @property
    def time_base(self):
        return self.frame.time_base

    @property
    def log_timestamps(self):
        return self.frame.log_timestamps

class AudioOutput(OutputFrame):
    frames: List[AudioFrame]
    request_id: str
    def __init__(self, frames: List[AudioFrame], request_id: str = ''):
        self.frames = frames
        self.request_id = request_id

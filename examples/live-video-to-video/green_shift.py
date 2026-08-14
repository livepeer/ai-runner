"""Green Shift -- minimal @pipeline example (function form).

Boosts the green channel of every video frame. No model download, no GPU
inference — just a tensor op. Ideal for verifying the pipeline infrastructure.

Usage:
    python examples/live-video-to-video/green_shift.py
"""

import torch

from runner.app import start_app
from runner.live.pipelines import pipeline, BaseParams
from runner.live.trickle import VideoFrame


@pipeline(name="green-shift")
async def green_shift(frame: VideoFrame, params: BaseParams) -> torch.Tensor:
    """Boost the green channel of every frame.

    Frame tensor layout: (B, H, W, C), values in [-1.0, 1.0].
    Channel order: R=0, G=1, B=2.
    """
    tensor = frame.tensor.clone()
    tensor[:, :, :, 1] = torch.clamp(tensor[:, :, :, 1] + 0.3, -1.0, 1.0)
    return tensor

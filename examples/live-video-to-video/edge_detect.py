"""Edge Detection (Sobel) -- @pipeline example (class form) with lifecycle hooks.

Pure-torch Sobel edge detection. No external model needed.
Demonstrates on_ready, transform, on_update, and on_stop lifecycle hooks.

Usage:
    python examples/live-video-to-video/edge_detect.py
"""

import logging

import torch
import torch.nn.functional as F
from pydantic import Field

from runner.app import start_app
from runner.live.pipelines import pipeline, BaseParams
from runner.live.trickle import VideoFrame


class EdgeParams(BaseParams):
    """Parameters for edge detection, adjustable mid-stream."""

    threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Edge detection threshold. Higher = fewer edges.",
    )
    colorize: bool = Field(
        default=False,
        description="Colorize edges based on gradient direction.",
    )


@pipeline(name="edge-detect", params=EdgeParams)
class EdgeDetect:
    """Real-time Sobel edge detection.

    Demonstrates:
    - on_ready: initializes Sobel kernels on the correct device
    - transform: runs edge detection per frame
    - on_update: adjusts threshold mid-stream
    - on_stop: cleanup on shutdown
    """

    def on_ready(self, **params):
        self._threshold = params.get("threshold", 0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sobel kernels for horizontal and vertical gradients
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)

        logging.info(f"EdgeDetect ready on {self.device}")

    def transform(self, frame: VideoFrame, params: EdgeParams) -> torch.Tensor:
        """Run Sobel edge detection on each frame."""
        # frame.tensor is (B, H, W, C) in [-1.0, 1.0]
        tensor = frame.tensor.to(self.device)

        # Convert to grayscale: (B, H, W, C) -> (B, 1, H, W)
        gray = tensor.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)

        # Apply Sobel filters
        edges_x = F.conv2d(gray, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray, self.sobel_y, padding=1)

        # Edge magnitude
        magnitude = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        magnitude = magnitude / (magnitude.max() + 1e-8)  # normalize to [0, 1]

        # Apply threshold
        edges = (magnitude > self._threshold).float()

        if params.colorize:
            # Color edges by gradient direction
            angle = torch.atan2(edges_y, edges_x + 1e-8)  # [-pi, pi]
            angle = (angle + torch.pi) / (2 * torch.pi)    # [0, 1]
            r = edges * angle
            g = edges * (1.0 - torch.abs(angle - 0.5) * 2.0)
            b = edges * (1.0 - angle)
            out = torch.cat([r, g, b], dim=1)  # (B, 3, H, W)
        else:
            out = edges.expand(-1, 3, -1, -1)  # (B, 3, H, W)

        # Convert back to (B, H, W, C) in [-1, 1]
        out = out.permute(0, 2, 3, 1)
        out = out * 2.0 - 1.0
        return out

    def on_update(self, **params):
        """Update threshold when params change mid-stream."""
        self._threshold = params.get("threshold", 0.1)
        logging.info(f"Edge threshold updated: {self._threshold}")

    def on_stop(self):
        logging.info("EdgeDetect stopped")

"""Example pipeline implementation."""

import asyncio
import logging

from app.live.pipelines.interface import Pipeline
from app.live.pipelines.trickle import VideoFrame, VideoOutput

logger = logging.getLogger(__name__)


class ExamplePipeline(Pipeline):
    """Example pipeline that passes through frames unchanged."""

    def __init__(self):
        super().__init__()
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.initialized = False

    async def initialize(self, **params):
        """Initialize the pipeline with parameters."""
        logger.info(f"Initializing ExamplePipeline with params: {params}")
        self.initialized = True
        logger.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        """Process an input frame."""
        # Simple pass-through example
        output = VideoOutput(frame, request_id)
        await self.frame_queue.put(output)

    async def get_processed_video_frame(self) -> VideoOutput:
        """Get the next processed frame."""
        return await self.frame_queue.get()

    async def update_params(self, **params):
        """Update pipeline parameters."""
        logger.info(f"Updating params: {params}")
        # Return None if no reload needed, or a Task if reload is required
        return None

    async def stop(self):
        """Clean up resources."""
        self.frame_queue = asyncio.Queue()
        self.initialized = False

    @classmethod
    def prepare_models(cls):
        """Download/prepare models if needed."""
        logger.info("ExamplePipeline does not require model preparation")


import logging
import asyncio

from ..interface import Pipeline
from ...trickle import VideoFrame, VideoOutput


class Scope(Pipeline):
    def __init__(self):
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        await self.frame_queue.put(VideoOutput(frame, request_id))

    async def get_processed_video_frame(self) -> VideoOutput:
        out = await self.frame_queue.get()
        return out.replace_tensor(out.tensor.clone())

    async def initialize(self, **params):
        logging.info(f"Initializing Scope pipeline with params: {params}")
        # Verify scope packages are available at runtime
        try:
            from lib.schema import HealthResponse
            # Test that the import works by checking the class exists
            assert HealthResponse is not None
            logging.info("Successfully imported scope packages (lib.schema.HealthResponse)")
        except ImportError as e:
            logging.error(f"Failed to import scope packages during initialization: {e}")
            raise RuntimeError(f"Scope packages not available: {e}")
        logging.info("Pipeline initialization complete")

    async def update_params(self, **params):
        logging.info(f"Updating params: {params}")

    async def stop(self):
        logging.info("Stopping pipeline")
        # clear the frame queue
        self.frame_queue = asyncio.Queue()


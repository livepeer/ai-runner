from PIL import Image
from abc import ABC, abstractmethod
from trickle import VideoFrame, VideoOutput

class Pipeline(ABC):
    """Abstract base class for image processing pipelines.

    Processes frames sequentially and supports dynamic parameter updates.

    Notes:
    - Methods are only called one at a time in a separate process, so no need
      for any locking.
    - Error handling is done by the caller, so the implementation can let
      exceptions propagate for optimal error reporting.
    """

    def __init__(self):
        """Initialize pipeline with optional parameters."""
        pass

    @abstractmethod
    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        """Put a frame into the pipeline.

        Args:
            frame: Input VideoFrame
        """
        pass

    @abstractmethod
    async def get_processed_video_frame(self) -> VideoOutput:
        """Get a processed frame from the pipeline.

        Returns:
            Processed VideoFrame
        """
        pass

    @abstractmethod
    async def initialize(self, **params):
        """Initialize the pipeline with parameters and warm up the processing.

        This method sets up the initial pipeline state and performs warmup operations.
        Must maintain valid state on success or restore previous state on failure.
        Starts the pipeline loops in comfystream.

        Args:
            **params: Implementation-specific parameters
        """
        pass

    @abstractmethod
    async def update_params(self, **params):
        """Update pipeline parameters.

        Must maintain valid state on success or restore previous state on failure.
        Called sequentially with process_frame so concurrency is not an issue.

        Args:
            **params: Implementation-specific parameters
        """
        pass

    async def stop(self):
        """Stop the pipeline.

        Called once when the pipeline is no longer needed.
        """
        pass

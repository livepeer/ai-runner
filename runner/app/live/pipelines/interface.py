from asyncio import Task
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from trickle import VideoFrame, VideoOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT


class BaseParams(BaseModel):
    """
    Base parameters common to all pipelines.

    Includes shared image dimensions and UI behavior used by the orchestrating
    process (e.g., loading overlay while (re)initializing pipelines).
    """

    width: int = Field(
        default=DEFAULT_WIDTH,
        ge=384,
        le=1024,
        multiple_of=64,
        description="Output image width in pixels. Must be divisible by 64 and between 384-1024.",
    )

    height: int = Field(
        default=DEFAULT_HEIGHT,
        ge=384,
        le=1024,
        multiple_of=64,
        description="Output image height in pixels. Must be divisible by 64 and between 384-1024.",
    )

    show_reloading_frame: bool = Field(
        default=True,
        description="Whether to render a loading overlay while the pipeline initializes or reloads.",
    )

class Pipeline(ABC):
    """Abstract base class for image processing pipelines.

    Processes frames sequentially and supports dynamic parameter updates.

    Notes:
    - Error handling is done by the caller, so the implementation can let
      exceptions propagate for optimal error reporting.
    """

    def __init__(self):
        pass

    @abstractmethod
    async def initialize(self, **params):
        """Initialize the pipeline with parameters and warm up the processing.

        This method sets up the initial pipeline state and performs warmup operations.
        Must maintain valid state on success or restore previous state on failure.

        Args:
            **params: Implementation-specific parameters
        """
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
    async def update_params(self, **params) -> Task[None] | None:
        """Update pipeline parameters.

        Must maintain valid state on success or restore previous state on failure.
        Called sequentially with process_frame so concurrency is not an issue.

        If the update will take a long time (e.g. reloading the pipeline),
        return a Task that will be awaited by the caller.

        Args:
            **params: Implementation-specific parameters
        """
        pass

    async def stop(self):
        """Stop the pipeline.

        Called once when the pipeline is no longer needed.
        """
        pass

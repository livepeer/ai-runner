"""Example pipeline parameters."""

from app.live.pipelines.interface import BaseParams
from pydantic import Field


class ExamplePipelineParams(BaseParams):
    """Parameters for ExamplePipeline."""

    # Add custom parameters here
    # BaseParams already provides: width, height, show_reloading_frame

    example_param: str = Field(
        default="default_value",
        description="An example parameter"
    )


from typing import List, Literal

from pydantic import BaseModel, Field

from ..interface import BaseParams


class PromptConfig(BaseModel):
    """Configuration for a single prompt with weight."""

    text: str
    weight: int = 100


class ScopeParams(BaseParams):
    """
    Scope pipeline parameters for longlive text-to-video generation.
    """

    pipeline: Literal["longlive"]
    """The scope pipeline to use. Currently only 'longlive' is supported."""

    prompts: List[PromptConfig] = Field(
        default_factory=lambda: [
            PromptConfig(
                text="A flying dragon soaring towards the camera, wings spread wide, scales glistening",
                weight=100,
            )
        ]
    )
    """List of prompts with weights for generation."""

    seed: int = 42
    """Random seed for reproducible generation."""


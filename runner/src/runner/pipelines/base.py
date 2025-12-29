from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastapi import APIRouter

class HealthCheck(BaseModel):
    status: Literal["LOADING", "OK", "ERROR", "IDLE"] = Field(..., description="The health status of the pipeline")

class Version(BaseModel):
    pipeline: str
    model_id: str
    version: str = Field(..., description="The version of the Runner")

class Pipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The pipeline name used for routing (e.g. 'text-to-image', 'live-video-to-video', etc)."""
        ...

    @abstractmethod
    def __init__(self, model_id: str, model_dir: str):
        self.model_id: str # declare the field here so the type hint is available when using this abstract class
        raise NotImplementedError("Pipeline should implement an __init__ method")

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        raise NotImplementedError("Pipeline should implement a __call__ method")

    def get_health(self) -> HealthCheck:
        """
        Returns a health check object for the pipeline.
        """
        return HealthCheck(status="OK", version="undefined")

    @property
    def router(self) -> "APIRouter | None":
        """
        Returns the API router for this pipeline. Override in subclasses.
        Uses lazy import to avoid circular dependencies.
        """
        return None

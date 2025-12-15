"""Pipeline specification for dynamically loading pipelines and their parameters classess.

Import Path Format:
    Classes are specified as "module.path:ClassName" where the colon separates
    the dotted module path from the class name. This allows pipelines and their
    params classes to be loaded dynamically at runtime.
"""

from typing import cast

from pydantic import BaseModel, model_validator


class PipelineSpec(BaseModel):
    """Specification for dynamically loading a pipeline and its parameters class.

    Import paths use "module.path:ClassName" format where the colon separates
    the dotted module path from the class name.

    e.g.:
        >>> spec = PipelineSpec(
        ...     pipeline_cls="my_package.pipeline:CustomPipeline",
        ...     params_cls="my_package.params:CustomParams",
        ...     initial_params={"strength": 0.8}
        ... )
    """
    name: str = ""
    """
    Identifier for the pipeline. Derived from pipeline_cls if not provided.
    This must match the MODEL_ID set by the Orchestrator for the runner container.
    """

    pipeline_cls: str
    """Import path to the Pipeline subclass. e.g. "my_pipelines.module:MyPipeline" """

    params_cls: str | None = None
    """Import path to the BaseParams subclass, or None to use generic BaseParams."""

    initial_params: dict = {}
    """Default parameter values passed to the pipeline on init."""

    @model_validator(mode="before")
    @classmethod
    def _set_default_name(cls, values: dict) -> dict:
        if not values.get("name") and "pipeline_cls" in values:
            name = cast(str, values["pipeline_cls"])
            name = name.rsplit(":", 1)[-1].lower().removesuffix("pipeline")
            values["name"] = name
        return values


def builtin_pipeline_spec(name: str) -> PipelineSpec | None:
    """
    Look up a built-in pipeline by name and return a PipelineSpec if found.
    """
    if name == "comfyui":
        return PipelineSpec(
            name="comfyui",
            pipeline_cls="app.live.pipelines.comfyui.pipeline:ComfyUI",
            params_cls="app.live.pipelines.comfyui.params:ComfyUIParams",
        )
    if name == "scope":
        return PipelineSpec(
            name="scope",
            pipeline_cls="app.live.pipelines.scope.pipeline:Scope",
            params_cls="app.live.pipelines.scope.params:ScopeParams",
        )
    if name == "noop":
        return PipelineSpec(
            name="noop",
            pipeline_cls="app.live.pipelines.noop:Noop",
        )
    return None

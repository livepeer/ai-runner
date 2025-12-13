"""Pipeline loader utilities for dynamic pipeline and params class loading.

Import Path Format:
    Classes are specified as "module.path:ClassName" where the colon separates
    the dotted module path from the class name. This allows pipelines and their
    params classes to be loaded dynamically at runtime.
"""

import sys
import time
from contextlib import contextmanager
import importlib
from typing import cast

from pydantic import BaseModel, model_validator

from .interface import Pipeline, BaseParams


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
    name: str | None = None
    """Identifier for the pipeline. Derived from pipeline_cls if not provided."""

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
    if name == "streamdiffusion" or name.startswith("streamdiffusion-"):
        return PipelineSpec(
            name="streamdiffusion",
            pipeline_cls="live.pipelines.streamdiffusion.pipeline:StreamDiffusion",
            params_cls="live.pipelines.streamdiffusion.params:StreamDiffusionParams",
        )
    if name == "comfyui":
        return PipelineSpec(
            name="comfyui",
            pipeline_cls="live.pipelines.comfyui.pipeline:ComfyUI",
            params_cls="live.pipelines.comfyui.params:ComfyUIParams",
        )
    elif name == "scope":
        return PipelineSpec(
            name="scope",
            pipeline_cls="live.pipelines.scope.pipeline:Scope",
            params_cls="live.pipelines.scope.params:ScopeParams",
        )
    elif name == "noop":
        return PipelineSpec(
            name="noop",
            pipeline_cls="live.pipelines.noop.pipeline:Noop",
        )
    else:
        return None


def load_pipeline_class(pipeline_cls: str) -> type:
    """Dynamically import and return a pipeline class.

    Args:
        pipeline_cls: Import path in the format "module.path:ClassName".
            The colon separates the module path (dotted Python import path)
            from the class name. e.g.: "custom_pipeline.package:CustomPipeline"
    """
    return _import_class(pipeline_cls)

def load_pipeline(pipeline_spec: PipelineSpec) -> Pipeline:
    """Load and instantiate a pipeline from its specification.

    Args:
        pipeline_spec: Specification containing the pipeline class import path.

    Returns:
        A new instance of the pipeline.
    """
    pipeline_class = load_pipeline_class(pipeline_spec.pipeline_cls)
    return pipeline_class()

def parse_pipeline_params(spec: PipelineSpec, params: dict) -> BaseParams:
    """Parse and validate pipeline parameters using the spec's params class.

    This function may be called from outside the pipeline process, so it guards
    against accidentally importing expensive libraries (torch, streamdiffusion,
    comfystream) during parameter parsing.

    Args:
        spec: Pipeline specification. If params_cls is set, it will be used to
            parse params; uses the same "module.path:ClassName" import format
            as pipeline_cls. If params_cls is None, returns a BaseParams instance.
        params: Dictionary of parameter values to parse.
    """
    with _no_expensive_imports():
        if spec.params_cls is None:
            return BaseParams(**params)

        params_class = _import_class(spec.params_cls)
        return params_class(**params)


@contextmanager
def _no_expensive_imports(timeout: float = 0.5):
    """Context manager to ensure no expensive modules are imported and import is fast."""
    expensive_modules = ("torch", "streamdiffusion", "comfystream")
    expensive_before = {m for m in expensive_modules if m in sys.modules}
    start_time = time.perf_counter()

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        expensive_after = {m for m in expensive_modules if m in sys.modules}

        if elapsed > timeout:
            raise TimeoutError(
                f"Import took {elapsed:.3f}s, exceeded timeout of {timeout}s. "
                "This likely indicates an expensive library is being imported."
            )

        if expensive_after - expensive_before:
            raise ImportError(
                f"Import imported expensive modules: {expensive_after - expensive_before}"
            )

def _import_class(import_path: str) -> type:
    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

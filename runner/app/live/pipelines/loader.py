import sys
import time
from contextlib import contextmanager
import importlib

from pydantic import BaseModel

from .interface import Pipeline, BaseParams

class PipelineSpec(BaseModel):
    """
    Specification for the classes to use for a pipeline and its parameters.
    """
    name: str
    pipeline_cls: str
    params_cls: str | None = None
    initial_params: dict = {}

    def __init__(self, name: str, pipeline_cls: str, params_cls: str | None = None, initial_params: dict = {}):
        super().__init__(name=name, pipeline_cls=pipeline_cls, params_cls=params_cls, initial_params=initial_params)


def builtin_pipeline_spec(name: str) -> PipelineSpec | None:
    if name == "streamdiffusion" or name.startswith("streamdiffusion-"):
        return PipelineSpec("streamdiffusion", "live.pipelines.streamdiffusion.pipeline:StreamDiffusion", "live.pipelines.streamdiffusion.params:StreamDiffusionParams")
    if name == "comfyui":
        return PipelineSpec("comfyui", "live.pipelines.comfyui.pipeline:ComfyUI", "live.pipelines.comfyui.params:ComfyUIParams")
    elif name == "scope":
        return PipelineSpec("scope", "live.pipelines.scope.pipeline:Scope", "live.pipelines.scope.params:ScopeParams")
    elif name == "noop":
        return PipelineSpec("noop", "live.pipelines.noop.pipeline:Noop")
    else:
        return None


def load_pipeline(pipeline_spec: PipelineSpec) -> Pipeline:
    pipeline_class = _import_class(pipeline_spec.pipeline_cls)
    return pipeline_class()


def parse_pipeline_params(spec: PipelineSpec, params: dict) -> BaseParams:
    """
    Parse pipeline parameters. This function may be called from outside the
    pipeline process, so we need to ensure no expensive libraries are imported.
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

"""
Pipeline loader utilities for dynamic pipeline and params class loading.
"""

import sys
import time
from contextlib import contextmanager
import importlib

from .interface import Pipeline, BaseParams
from .spec import PipelineSpec

def load_pipeline_class(pipeline_cls: str) -> type[Pipeline]:
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

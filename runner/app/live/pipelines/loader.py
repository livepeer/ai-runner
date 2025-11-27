import sys
import time
import logging
from contextlib import contextmanager
from importlib.metadata import entry_points, EntryPoint
from typing import Optional

from .interface import Pipeline, BaseParams

logger = logging.getLogger(__name__)

# Cache for discovered entry points (store EntryPoint objects, not loaded classes)
_pipeline_entry_points: Optional[dict[str, EntryPoint]] = None
_params_entry_points: Optional[dict[str, EntryPoint]] = None


def _discover_entry_points() -> tuple[dict[str, EntryPoint], dict[str, EntryPoint]]:
    """Discover pipeline and params entry points from installed packages.

    Returns tuple of (pipeline_eps, params_eps) - EntryPoint objects for lazy loading.
    """
    global _pipeline_entry_points, _params_entry_points

    if _pipeline_entry_points is not None and _params_entry_points is not None:
        return _pipeline_entry_points, _params_entry_points

    _pipeline_entry_points = {}
    _params_entry_points = {}

    try:
        for ep in entry_points(group="ai_runner.pipeline"):
            _pipeline_entry_points[ep.name] = ep
            logger.info(f"Discovered pipeline: {ep.name} -> {ep.value}")

        for ep in entry_points(group="ai_runner.pipeline_params"):
            _params_entry_points[ep.name] = ep
            logger.info(f"Discovered params: {ep.name} -> {ep.value}")
    except Exception as e:
        logger.warning(f"Error discovering entry points: {e}")
        if _pipeline_entry_points is None:
            _pipeline_entry_points = {}
        if _params_entry_points is None:
            _params_entry_points = {}

    return _pipeline_entry_points, _params_entry_points


def _get_pipeline_entry_point(name: str) -> Optional[EntryPoint]:
    """Get pipeline entry point by name, trying exact match then base name for variants."""
    base_name = name.split("-")[0] if "-" in name else name
    pipeline_eps, _ = _discover_entry_points()
    return pipeline_eps.get(name) or pipeline_eps.get(base_name)


def _get_params_entry_point(name: str) -> Optional[EntryPoint]:
    """Get params entry point by name, trying exact match then base name for variants."""
    base_name = name.split("-")[0] if "-" in name else name
    _, params_eps = _discover_entry_points()
    return params_eps.get(name) or params_eps.get(base_name)


def load_pipeline(name: str) -> Pipeline:
    """Load a pipeline by name from entry points.

    For names like "streamdiffusion-sd15", it will try:
    1. Entry point "streamdiffusion-sd15" (exact match)
    2. Entry point "streamdiffusion" (base name for variants)
    """
    ep = _get_pipeline_entry_point(name)

    if ep:
        logger.info(f"Loading pipeline '{name}' from entry point: {ep.value}")
        pipeline_class = ep.load()
        return pipeline_class()

    pipeline_eps, _ = _discover_entry_points()
    raise ValueError(
        f"Unknown pipeline: {name}. "
        f"Available pipelines: {', '.join(sorted(pipeline_eps.keys()))}"
    )


def parse_pipeline_params(name: str, params: dict) -> BaseParams:
    """Parse pipeline parameters WITHOUT importing the pipeline class.

    This function may be called from outside the pipeline process, so we need
    to ensure no expensive libraries (torch, etc.) are imported.

    Uses the `ai_runner.pipeline_params` entry point group for explicit params discovery.
    Falls back to `BaseParams` if no params entry point exists for this pipeline.
    """
    # First check if pipeline exists at all
    pipeline_ep = _get_pipeline_entry_point(name)
    if not pipeline_ep:
        pipeline_eps, _ = _discover_entry_points()
        raise ValueError(
            f"Unknown pipeline: {name}. "
            f"Available pipelines: {', '.join(sorted(pipeline_eps.keys()))}"
        )

    # Try to get params entry point
    params_ep = _get_params_entry_point(name)

    if params_ep:
        with _no_expensive_imports():
            logger.debug(f"Loading params for '{name}' from entry point: {params_ep.value}")
            params_class = params_ep.load()
            return params_class(**params)
    else:
        # No params entry point - use BaseParams
        logger.debug(f"No params entry point for '{name}', using BaseParams")
        return BaseParams(**params)


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

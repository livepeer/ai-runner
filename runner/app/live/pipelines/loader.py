import sys
import time
import logging
from contextlib import contextmanager
from typing import Optional, Type
from importlib.metadata import entry_points

from .interface import Pipeline, BaseParams

logger = logging.getLogger(__name__)

# Cache for discovered entry points
_pipeline_entry_points: Optional[dict[str, Type[Pipeline]]] = None


def _discover_entry_points():
    """Discover pipeline entry points from installed packages."""
    global _pipeline_entry_points

    if _pipeline_entry_points is not None:
        return _pipeline_entry_points

    _pipeline_entry_points = {}

    try:
        # Discover pipeline entry points
        for ep in entry_points(group="ai_runner.pipelines"):
            try:
                pipeline_class = ep.load()
                _pipeline_entry_points[ep.name] = pipeline_class
                logger.info(f"Discovered pipeline entry point: {ep.name} from {ep.value}")
            except Exception as e:
                logger.warning(f"Failed to load pipeline entry point {ep.name}: {e}")
    except Exception as e:
        logger.warning(f"Error discovering entry points: {e}")
        # Fall back to empty dict if entry_points() fails (Python < 3.10)
        if _pipeline_entry_points is None:
            _pipeline_entry_points = {}

    return _pipeline_entry_points


def load_pipeline(name: str) -> Pipeline:
    """
    Load a pipeline by name from entry points.

    For names like "streamdiffusion-sd15", it will try:
    1. Entry point "streamdiffusion-sd15" (exact match)
    2. Entry point "streamdiffusion" (base name for variants)
    """
    # Normalize pipeline name (for variants like "streamdiffusion-sd15")
    base_name = name.split("-")[0] if "-" in name else name

    pipeline_eps = _discover_entry_points()

    # Try exact match first
    if name in pipeline_eps:
        logger.info(f"Loading pipeline '{name}' from entry point")
        return pipeline_eps[name]()

    # Try base name for variants
    if base_name in pipeline_eps:
        logger.info(f"Loading pipeline '{name}' using base entry point '{base_name}'")
        return pipeline_eps[base_name]()

    raise ValueError(
        f"Unknown pipeline: {name}. "
        f"Available pipelines: {', '.join(sorted(pipeline_eps.keys()))}"
    )


def parse_pipeline_params(name: str, params: dict) -> BaseParams:
    """
    Parse pipeline parameters. This function may be called from outside the
    pipeline process, so we need to ensure no expensive libraries are imported.

    Gets the params class from the Pipeline class itself (via Pipeline.Params).
    """
    # Normalize pipeline name
    base_name = name.split("-")[0] if "-" in name else name

    # Get pipeline class from entry points
    pipeline_eps = _discover_entry_points()

    pipeline_class: Optional[Type[Pipeline]] = None

    # Try exact match first
    if name in pipeline_eps:
        pipeline_class = pipeline_eps[name]
    # Try base name for variants
    elif base_name in pipeline_eps:
        pipeline_class = pipeline_eps[base_name]

    if pipeline_class:
        with _no_expensive_imports():
            params_class = pipeline_class.Params
            return params_class(**params)

    raise ValueError(
        f"Unknown pipeline: {name}. "
        f"Available pipelines: {', '.join(sorted(pipeline_eps.keys()))}"
    )


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

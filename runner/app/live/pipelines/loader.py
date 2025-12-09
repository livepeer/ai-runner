import importlib
import logging
from typing import Type

from .interface import Pipeline, BaseParams

logger = logging.getLogger(__name__)

# Default import paths for built-in noop pipeline
DEFAULT_PIPELINE_IMPORT = "app.live.pipelines.noop:Noop"
DEFAULT_PARAMS_IMPORT = ""  # Empty means use BaseParams


def _import_class(import_path: str) -> Type:
    """Import and return a class from a path like 'app.live.pipelines.noop:Noop'.

    Args:
        import_path: Full import path in format 'module.path:ClassName'

    Returns:
        The imported class

    Raises:
        ValueError: If import path format is invalid
        ImportError: If module or class cannot be found
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid import path format: {import_path}. "
            "Expected format: 'module.path:ClassName'"
        )

    module_path, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_pipeline(pipeline_import: str = "") -> Pipeline:
    """Load a pipeline from its import path.

    Args:
        pipeline_import: Full import path like 'app.live.pipelines.noop:Noop'.
                        If empty, loads the default noop pipeline.

    Returns:
        An instance of the Pipeline class
    """
    import_path = pipeline_import or DEFAULT_PIPELINE_IMPORT
    logger.info(f"Loading pipeline from import path: {import_path}")

    pipeline_class = _import_class(import_path)
    return pipeline_class()


def parse_pipeline_params(params_import: str, params: dict) -> BaseParams:
    """Parse pipeline parameters using the specified params class.

    This function can be called from outside the pipeline process, so we need
    to ensure no expensive libraries (torch, etc.) are imported when using
    lightweight params classes.

    Args:
        params_import: Full import path like 'app.live.pipelines.streamdiffusion.params:StreamDiffusionParams'.
                      If empty, uses BaseParams.
        params: Dictionary of parameter values

    Returns:
        An instance of the params class populated with the provided values
    """
    if not params_import:
        logger.debug("No params import specified, using BaseParams")
        return BaseParams(**params)

    logger.debug(f"Loading params from import path: {params_import}")
    params_class = _import_class(params_import)
    return params_class(**params)

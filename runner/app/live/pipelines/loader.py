import sys
import time
from contextlib import contextmanager
from .interface import Pipeline, BaseParams


def load_pipeline(name: str) -> Pipeline:
    if name == "streamdiffusion" or name.startswith("streamdiffusion-"):
        from .streamdiffusion.pipeline import StreamDiffusion
        return StreamDiffusion()
    if name == "comfyui":
        from .comfyui.pipeline import ComfyUI
        return ComfyUI()
    elif name == "scope":
        from .scope.pipeline import Scope
        return Scope()
    elif name == "noop":
        from .noop import Noop
        return Noop()
    raise ValueError(f"Unknown pipeline: {name}")


def parse_pipeline_params(name: str, params: dict) -> BaseParams:
    """
    Parse pipeline parameters. This function may be called from outside the
    pipeline process, so we need to ensure no expensive libraries are imported.
    """
    if name == "streamdiffusion" or name.startswith("streamdiffusion-"):
        with _no_expensive_imports():
            from .streamdiffusion.params import StreamDiffusionParams
            return StreamDiffusionParams(**params)
    if name == "comfyui":
        with _no_expensive_imports():
            from .comfyui.params import ComfyUIParams
            return ComfyUIParams(**params)
    if name == "scope":
        with _no_expensive_imports():
            from .scope.params import ScopeParams
            return ScopeParams(**params)
    if name == "noop":
        return BaseParams(**params)
    raise ValueError(f"Unknown pipeline: {name}")


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

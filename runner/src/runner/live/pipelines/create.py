"""@pipeline decorator for creating live pipelines.

Lifecycle hooks:
* ``prepare_models``: called at build time (model download, TensorRT compile)
* ``on_ready``: called once at startup
* ``transform``: called per frame
* ``on_update``: called when params change mid-stream
* ``on_stop``: called on shutdown
"""

import asyncio
import inspect
import logging
from typing import Optional, Type

from .interface import Pipeline, BaseParams
from .spec import PipelineSpec
from ..trickle import VideoFrame, VideoOutput


async def _invoke(func, *args, **kwargs):
    """Call a function, handling both async and sync. Sync runs in a thread pool."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def pipeline(
    name: str,
    params: Optional[Type[BaseParams]] = None,
    initial_params: Optional[dict] = None,
):
    """Decorator to define a pipeline. Can decorate a function or class.

    Args:
        name: Pipeline identifier. Must match the MODEL_ID on the orchestrator.
        params: Pydantic model for pipeline parameters. Defaults to BaseParams.
        initial_params: Default parameter values passed on init.
    """
    params_cls = params or BaseParams

    def decorator(func_or_class):
        if callable(func_or_class) and not isinstance(func_or_class, type):
            # Function form: wrap into a minimal class with a transform method.
            func = func_or_class

            class _Wrapper:
                async def transform(self, frame, p):
                    return await _invoke(func, frame, p)

            _Wrapper.__name__ = func.__name__
            _Wrapper.__qualname__ = func.__qualname__
            _Wrapper.__module__ = func.__module__
            user_cls = _Wrapper
        elif isinstance(func_or_class, type):
            user_cls = func_or_class
        else:
            raise TypeError(
                "@pipeline can only decorate a function or class, got "
                f"{type(func_or_class)}"
            )

        if "." in user_cls.__qualname__:
            logging.warning(
                f"@pipeline decorating nested '{user_cls.__qualname__}' — "
                f"this may cause import issues if the enclosing scope is not accessible"
            )

        pipeline_cls = _build_pipeline(user_cls, params_cls)

        # Auto-generate PipelineSpec.
        params_import = (
            f"{params_cls.__module__}:{params_cls.__qualname__}"
            if params_cls is not BaseParams
            else None
        )
        pipeline_cls._spec = PipelineSpec(
            name=name,
            pipeline_cls=f"{pipeline_cls.__module__}:{pipeline_cls.__qualname__}",
            params_cls=params_import,
            initial_params=initial_params or {},
        )

        return pipeline_cls

    return decorator


def _build_pipeline(user_cls, params_cls: Type[BaseParams]) -> Type[Pipeline]:
    """Build a Pipeline subclass from a user class.

    The user class must define a ``transform`` method. Lifecycle hooks are
    detected by name: ``on_ready``, ``on_update``, ``on_stop``,
    ``prepare_models``.
    """
    if not hasattr(user_cls, "transform"):
        raise TypeError(
            f"@pipeline class {user_cls.__name__} must define a 'transform' method"
        )

    has_on_ready = hasattr(user_cls, "on_ready")
    has_on_update = hasattr(user_cls, "on_update")
    has_on_stop = hasattr(user_cls, "on_stop")
    has_prepare = hasattr(user_cls, "prepare_models")
    label = user_cls.__name__

    class GeneratedPipeline(Pipeline):
        def __init__(self):
            super().__init__()
            self._lock = asyncio.Lock()
            self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
            self.params_instance: Optional[BaseParams] = None
            self._inner = user_cls()

        async def initialize(self, **kw_params):
            logging.info(f"Initializing {label} pipeline with params: {kw_params}")
            async with self._lock:
                self.params_instance = params_cls(**kw_params)
                if has_on_ready:
                    await _invoke(self._inner.on_ready, **kw_params)
            logging.info("Pipeline initialization complete")

        async def put_video_frame(self, frame: VideoFrame, request_id: str):
            async with self._lock:
                result = await _invoke(
                    self._inner.transform, frame, self.params_instance
                )
            if isinstance(result, VideoOutput):
                # Normalize request_id to match the current stream.
                if result.request_id != request_id:
                    result = VideoOutput(result.frame, request_id)
                await self.frame_queue.put(result)
            else:
                await self.frame_queue.put(
                    VideoOutput(frame, request_id).replace_tensor(result)
                )

        async def get_processed_video_frame(self) -> VideoOutput:
            return await self.frame_queue.get()

        async def update_params(self, **kw_params):
            logging.info(f"Updating {label} params: {kw_params}")
            async with self._lock:
                self.params_instance = params_cls(**kw_params)
                if has_on_update:
                    await _invoke(self._inner.on_update, **kw_params)

        async def stop(self):
            logging.info(f"Stopping {label} pipeline")
            if has_on_stop:
                await _invoke(self._inner.on_stop)

        @classmethod
        async def prepare_models(cls):
            if has_prepare:
                await _invoke(user_cls.prepare_models)
            else:
                logging.info(f"{label} pipeline does not require model preparation")

    # Keep the original decorated name so importlib can find it via
    # getattr(module, name) after the decorator replaces the symbol.
    GeneratedPipeline.__name__ = user_cls.__name__
    GeneratedPipeline.__qualname__ = user_cls.__qualname__
    GeneratedPipeline.__module__ = user_cls.__module__

    return GeneratedPipeline

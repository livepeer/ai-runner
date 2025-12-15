import logging
import os
from contextlib import asynccontextmanager

from app.routes import health, hardware, version
from fastapi import FastAPI
from fastapi.routing import APIRoute
from app.utils.hardware import HardwareInfo
from app.live.log import config_logging
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from app.pipelines.base import Pipeline

config_logging(log_level=logging.DEBUG if os.getenv("VERBOSE_LOGGING")=="1" else logging.INFO)
logger = logging.getLogger(__name__)

VERSION = Gauge('version', 'Runner version', ['app', 'version'])

def _setup_app(app: FastAPI, pipeline: Pipeline):
    app.pipeline = pipeline
    # Create application wide hardware info service.
    app.hardware_info_service = HardwareInfo()

    app.include_router(health.router)
    app.include_router(hardware.router)
    app.include_router(version.router)

    if pipeline.router is None:
        raise NotImplementedError(f"{type(pipeline).__name__} does not have a router defined")
    app.include_router(pipeline.router)

    app.hardware_info_service.log_gpu_compute_info()


def load_pipeline(pipeline: str, model_id: str) -> Pipeline:
    match pipeline:
        case "text-to-image":
            from app.pipelines.text_to_image import TextToImagePipeline

            return TextToImagePipeline(model_id)
        case "image-to-image":
            from app.pipelines.image_to_image import ImageToImagePipeline

            return ImageToImagePipeline(model_id)
        case "image-to-video":
            from app.pipelines.image_to_video import ImageToVideoPipeline

            return ImageToVideoPipeline(model_id)
        case "audio-to-text":
            from app.pipelines.audio_to_text import AudioToTextPipeline

            return AudioToTextPipeline(model_id)
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            from app.pipelines.upscale import UpscalePipeline

            return UpscalePipeline(model_id)
        case "segment-anything-2":
            from app.pipelines.segment_anything_2 import SegmentAnything2Pipeline

            return SegmentAnything2Pipeline(model_id)
        case "llm":
            from app.pipelines.llm import LLMPipeline

            return LLMPipeline(model_id)
        case "image-to-text":
            from app.pipelines.image_to_text import ImageToTextPipeline

            return ImageToTextPipeline(model_id)
        case "live-video-to-video":
            from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
            from app.live.pipelines import builtin_pipeline_spec

            pipeline_spec = builtin_pipeline_spec(model_id)
            if pipeline_spec is None:
                raise EnvironmentError(f"Live pipeline {model_id} not found")

            return LiveVideoToVideoPipeline(pipeline_spec)
        case "text-to-speech":
            from app.pipelines.text_to_speech import TextToSpeechPipeline

            return TextToSpeechPipeline(model_id)
        case _:
            raise EnvironmentError(
                f"{pipeline} is not a valid pipeline for model {model_id}"
            )


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """
    Create a configured AI Runner FastAPI app.

    Args:
        pipeline: Pipeline instance to use. If None, loads from PIPELINE and MODEL_ID
                  environment variables using the built-in pipeline registry.

    Returns:
        Configured FastAPI application ready to be run with an ASGI server.

    Example:
        main.py:
            from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
            app = create_app(pipeline=LiveVideoToVideoPipeline("streamdiffusion"))

        And to run the app with uvicorn:
            uvicorn main:app --host 0.0.0.0 --port 8000
    """
    runner_version=os.getenv("VERSION", "undefined")
    VERSION.labels(app="ai-runner", version=runner_version).set(1)
    logger.info("Runner version: %s", runner_version)

    if pipeline is None:
        pipeline_name = os.getenv("PIPELINE", "")
        model_id = os.getenv("MODEL_ID", "")
        if pipeline_name != "" and model_id != "":
            pipeline = load_pipeline(pipeline_name, model_id)


    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            raise EnvironmentError("Pipeline must be provided or set through the PIPELINE and MODEL_ID environment variables")

        _setup_app(app, pipeline)
        logger.info(f"Started up with pipeline={type(pipeline).__name__} model_id={pipeline.model_id}")

        yield

        logger.info("Shutting down")

    app = FastAPI(lifespan=lifespan)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Expose Prometheus metrics."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def start_app(

    pipeline: Pipeline | None = None,
    host: str | None = None,
    port: int | None = None,
    **uvicorn_kwargs,
):
    """
    Create and start an AI Runner app. Blocks until shutdown.

    Args:
        pipeline: Pipeline instance. Defaults to loading from PIPELINE/MODEL_ID env vars.
        host: Host to bind to. Defaults to HOST env var or "0.0.0.0".
        port: Port to bind to. Defaults to PORT env var or 8000.
        **uvicorn_kwargs: Additional arguments passed to uvicorn.run()

    Example:
        from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
        start_app(pipeline=LiveVideoToVideoPipeline("streamdiffusion"), port=8080)
    """
    import uvicorn

    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "8000"))

    app = create_app(pipeline=pipeline)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)

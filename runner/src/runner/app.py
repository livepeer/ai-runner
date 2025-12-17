import logging
import os
import sys
from contextlib import asynccontextmanager

from runner.routes import health, hardware, version
from fastapi import FastAPI
from fastapi.routing import APIRoute
from runner.utils.hardware import HardwareInfo
from runner.live.log import config_logging
from runner.live.pipelines import PipelineSpec
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from runner.pipelines.base import Pipeline

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
            from runner.pipelines.text_to_image import TextToImagePipeline

            return TextToImagePipeline(model_id)
        case "image-to-image":
            from runner.pipelines.image_to_image import ImageToImagePipeline

            return ImageToImagePipeline(model_id)
        case "image-to-video":
            from runner.pipelines.image_to_video import ImageToVideoPipeline

            return ImageToVideoPipeline(model_id)
        case "audio-to-text":
            from runner.pipelines.audio_to_text import AudioToTextPipeline

            return AudioToTextPipeline(model_id)
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            from runner.pipelines.upscale import UpscalePipeline

            return UpscalePipeline(model_id)
        case "segment-anything-2":
            from runner.pipelines.segment_anything_2 import SegmentAnything2Pipeline

            return SegmentAnything2Pipeline(model_id)
        case "llm":
            from runner.pipelines.llm import LLMPipeline

            return LLMPipeline(model_id)
        case "image-to-text":
            from runner.pipelines.image_to_text import ImageToTextPipeline

            return ImageToTextPipeline(model_id)
        case "live-video-to-video":
            from runner.pipelines.live_video_to_video import LiveVideoToVideoPipeline
            from runner.live.pipelines import builtin_pipeline_spec

            pipeline_spec = builtin_pipeline_spec(model_id)
            if pipeline_spec is None:
                raise EnvironmentError(f"Live pipeline {model_id} not found")

            return LiveVideoToVideoPipeline(pipeline_spec)
        case "text-to-speech":
            from runner.pipelines.text_to_speech import TextToSpeechPipeline

            return TextToSpeechPipeline(model_id)
        case _:
            raise EnvironmentError(
                f"{pipeline} is not a valid pipeline for model {model_id}"
            )


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


def prepare_models(pipeline_spec: PipelineSpec) -> None:
    """Prepare models for a live pipeline (download, compile TensorRT engines, etc.)."""
    from .live.pipelines.loader import load_pipeline_class

    logger.info(f"Preparing models for pipeline: {pipeline_spec.name}")
    pipeline_class = load_pipeline_class(pipeline_spec.pipeline_cls)
    pipeline_class.prepare_models()
    logger.info("Model preparation complete")


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    """Create a FastAPI app for use with custom ASGI servers."""
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
    pipeline: Pipeline | PipelineSpec | None = None,
    host: str | None = None,
    port: int | None = None,
    **uvicorn_kwargs,
):
    """
    Primary entrypoint for AI Runner applications. Handles both running the server
    and preparing models based on environment configuration.

    For live pipelines, pass a PipelineSpec directly. The function will:
    - If PREPARE_MODELS=1 env var (or --prepare-models arg): prepare models and exit
    - Otherwise: wrap in LiveVideoToVideoPipeline and start the server

    Args:
        pipeline: Pipeline instance or PipelineSpec for live pipelines.
                  Defaults to loading from PIPELINE/MODEL_ID env vars.
        host: Host to bind to. Defaults to HOST env var or "0.0.0.0".
        port: Port to bind to. Defaults to PORT env var or 8000.
        **uvicorn_kwargs: Additional arguments passed to uvicorn.run()

    Example (live pipeline):
        from runner.app import start_app
        from runner.live.pipelines import PipelineSpec

        pipeline_spec = PipelineSpec(
            name="my-pipeline",
            pipeline_cls="pipeline.pipeline:MyPipeline",
            params_cls="pipeline.params:MyParams",
        )

        if __name__ == "__main__":
            start_app(pipeline=pipeline_spec)

        # Run normally:     python main.py
        # Prepare models:   PREPARE_MODELS=1 python main.py
    """
    # Handle PipelineSpec for live pipelines
    if isinstance(pipeline, PipelineSpec):
        # Check for model preparation mode
        if os.getenv("PREPARE_MODELS") == "1" or "--prepare-models" in sys.argv:
            prepare_models(pipeline)
            return

        # Wrap in LiveVideoToVideoPipeline for normal operation
        from .pipelines.live_video_to_video import LiveVideoToVideoPipeline
        pipeline = LiveVideoToVideoPipeline(pipeline)

    import uvicorn

    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "8000"))

    app = create_app(pipeline=pipeline)
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)

import logging
import os
from contextlib import asynccontextmanager

from app.routes import health, hardware
from fastapi import FastAPI
from fastapi.routing import APIRoute
from app.utils.hardware import HardwareInfo
from app.live.log import config_logging
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

config_logging()
logger = logging.getLogger(__name__)

VERSION = Gauge('version', 'Runner version', ['app', 'version'])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create application wide hardware info service.
    app.hardware_info_service = HardwareInfo()

    app.include_router(health.router)
    app.include_router(hardware.router)

    pipeline = os.environ["PIPELINE"]
    model_id = os.environ.get("MODEL_ID", "")
    if pipeline != "transformers" and not model_id:
        raise EnvironmentError(f"MODEL_ID must be set when using pipeline {pipeline}")

    app.pipeline = load_pipeline(pipeline, model_id)
    app.include_router(load_route(pipeline))

    app.hardware_info_service.log_gpu_compute_info()
    logger.info(f"Started up with pipeline {app.pipeline}")

    VERSION.labels(app="ai-runner", version=os.getenv("GIT_SHA", "unknown")).set(1)

    yield

    logger.info("Shutting down")


def load_pipeline(pipeline: str, model_id: str) -> any:
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

            return LiveVideoToVideoPipeline(model_id)
        case "text-to-speech":
            from app.pipelines.text_to_speech import TextToSpeechPipeline

            return TextToSpeechPipeline(model_id)
        case "transformers":
            return None
        case _:
            raise EnvironmentError(
                f"{pipeline} is not a valid pipeline for model {model_id}"
            )


def load_route(pipeline: str) -> any:
    match pipeline:
        case "text-to-image":
            from app.routes import text_to_image

            return text_to_image.router
        case "image-to-image":
            from app.routes import image_to_image

            return image_to_image.router
        case "image-to-video":
            from app.routes import image_to_video

            return image_to_video.router
        case "audio-to-text":
            from app.routes import audio_to_text

            return audio_to_text.router
        case "frame-interpolation":
            raise NotImplementedError("frame-interpolation pipeline not implemented")
        case "upscale":
            from app.routes import upscale

            return upscale.router
        case "segment-anything-2":
            from app.routes import segment_anything_2

            return segment_anything_2.router
        case "llm":
            from app.routes import llm

            return llm.router
        case "image-to-text":
            from app.routes import image_to_text

            return image_to_text.router
        case "live-video-to-video":
            from app.routes import live_video_to_video

            return live_video_to_video.router
        case "text-to-speech":
            from app.routes import text_to_speech

            return text_to_speech.router
        case "transformers":
            from app.routes import transformers

            return transformers.router
        case _:
            raise EnvironmentError(f"{pipeline} is not a valid pipeline")


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


app = FastAPI(lifespan=lifespan)

@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

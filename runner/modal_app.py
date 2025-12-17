import logging
import os
from pathlib import Path

from modal import Image, Secret, Stub, Volume, asgi_app, enter, method

from runner.main import config_logging, load_route, use_route_names_as_operation_ids
from runner.routes import health

stub = Stub("livepeer-ai-runner")
pipeline_image = (
    Image.from_registry("livepeer/ai-runner:latest")
    .workdir("/app")
    .env({"BFLOAT16": "true"})
)
api_image = Image.debian_slim(python_version="3.11").pip_install(
    "pydantic==2.6.1", "fastapi==0.109.2", "pillow"
)
downloader_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==0.20.2",
        "hf-transfer==0.1.4",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
)
models_volume = Volume.persisted("models")
models_path = Path("/models")

logger = logging.getLogger(__name__)

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"


@stub.function(
    image=downloader_image,
    volumes={models_path: models_volume},
    timeout=3600,
    secrets=[Secret.from_name("huggingface")],
)
def download_model(model_id: str):
    from huggingface_hub import snapshot_download

    try:
        # TODO: Handle case where there are no fp16 safetensors available
        allow_patterns = ["*unet.safetensors", "*.fp16.safetensors", "*.json", "*.txt"]
        ignore_patterns = [".onnx", ".onnx_data"]
        cache_dir = "/models"

        snapshot_download(
            model_id,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=os.environ.get("HF_TOKEN"),
        )
        logger.info(f"Downloaded model {model_id} to volume")
        models_volume.commit()
    except Exception:
        logger.exception(f"Failed to download model {model_id} to volume")
        raise


class Pipeline:
    def __init__(self, pipeline: str, model_id: str):
        self.pipeline = pipeline
        self.model_id = model_id

    @enter()
    def enter(self):
        from runner.app import load_pipeline

        model_id = self.model_id
        if SDXL_LIGHTNING_MODEL_ID in self.model_id:
            model_id = SDXL_LIGHTNING_MODEL_ID

        model_dir = "models--" + model_id.replace("/", "--")
        path = models_path / model_dir
        if not path.exists():
            models_volume.reload()

        if not path.exists():
            raise Exception(f"No model found at {path}")

        self.pipe = load_pipeline(self.pipeline, self.model_id)

    @method()
    def predict(self, **kwargs):
        return self.pipe(**kwargs)


@stub.cls(
    gpu="A10G",
    image=pipeline_image,
    memory=1024,
    volumes={models_path: models_volume},
    container_idle_timeout=5 * 60,
)
class A10G_Pipeline(Pipeline):
    pass


@stub.cls(
    gpu="A100",
    image=pipeline_image,
    memory=1024,
    volumes={models_path: models_volume},
    container_idle_timeout=5 * 60,
)
class A100_Pipeline(Pipeline):
    pass


# Wrap Pipeline for dependency injection in the runner FastAPI route
class RunnerPipeline:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.model_id = pipeline.model_id

    def __call__(self, **kwargs):
        return self.pipeline.predict.remote(**kwargs)


def make_api(pipeline: str, model_id: str, gpu: str = "A10G"):
    from fastapi import FastAPI

    config_logging()

    app = FastAPI()

    app.include_router(health.router)

    if gpu == "A10G":
        app.pipeline = RunnerPipeline(A10G_Pipeline(pipeline, model_id))
    elif gpu == "A100":
        app.pipeline = RunnerPipeline(A100_Pipeline(pipeline, model_id))
    else:
        raise Exception(f"invalid gpu value {gpu}")

    app.include_router(load_route(pipeline))

    use_route_names_as_operation_ids(app)

    return app


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_lightning_api():
    return make_api("text-to-image", "ByteDance/SDXL-Lightning")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_lightning_4step_api():
    return make_api("text-to-image", "ByteDance/SDXL-Lightning-4step")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_lightning_8step_api():
    return make_api("text-to-image", "ByteDance/SDXL-Lightning-8step")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def image_to_image_sdxl_lightning_api():
    return make_api("image-to-image", "ByteDance/SDXL-Lightning")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def image_to_image_sdxl_lightning_4step_api():
    return make_api("image-to-image", "ByteDance/SDXL-Lightning-4step")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def image_to_image_sdxl_lightning_8step_api():
    return make_api("image-to-image", "ByteDance/SDXL-Lightning-8step")


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def text_to_image_sdxl_turbo_api():
    return make_api("text-to-image", "stabilityai/sdxl-turbo")


# @stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
# @asgi_app()
# def image_to_video_svd_api():
#     return make_api(
#         "image-to-video", "stabilityai/stable-video-diffusion-img2vid-xt", "A100"
#     )


@stub.function(image=api_image, secrets=[Secret.from_name("api-auth-token")])
@asgi_app()
def image_to_video_svd_1_1_api():
    return make_api(
        "image-to-video", "stabilityai/stable-video-diffusion-img2vid-xt-1-1", "A100"
    )

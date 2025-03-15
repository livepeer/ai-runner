import asyncio
import logging
import os
import random
from typing import Annotated, Dict, Tuple, Union

import torch
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    ImageResponse,
    handle_pipeline_exception,
    http_error,
    image_to_data_url,
)

router = APIRouter()

logger = logging.getLogger(__name__)


# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing output image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}


class TextToImageParams(BaseModel):
    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        str,
        Field(
            default="", description="Hugging Face model ID used for image generation."
        ),
    ]
    loras: Annotated[
        str,
        Field(
            default="",
            description=(
                "A LoRA (Low-Rank Adaptation) model and its corresponding weight for "
                'image generation. Example: { "latent-consistency/lcm-lora-sdxl": '
                '1.0, "nerijs/pixel-art-xl": 1.2}.'
            ),
        ),
    ]
    prompt: Annotated[
        str,
        Field(
            description=(
                "Text prompt(s) to guide image generation. Separate multiple prompts "
                "with '|' if supported by the model."
            )
        ),
    ]
    height: Annotated[
        int,
        Field(default=576, description="The height in pixels of the generated image."),
    ]
    width: Annotated[
        int,
        Field(default=1024, description="The width in pixels of the generated image."),
    ]
    guidance_scale: Annotated[
        float,
        Field(
            default=7.5,
            description=(
                "Encourages model to generate images closely linked to the text prompt "
                "(higher values may reduce image quality)."
            ),
        ),
    ]
    negative_prompt: Annotated[
        str,
        Field(
            default="",
            description=(
                "Text prompt(s) to guide what to exclude from image generation. "
                "Ignored if guidance_scale < 1."
            ),
        ),
    ]
    safety_check: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Perform a safety check to estimate if generated images could be "
                "offensive or harmful."
            ),
        ),
    ]
    seed: Annotated[
        int, Field(default=None, description="Seed for random number generation.")
    ]
    num_inference_steps: Annotated[
        int,
        Field(
            default=50,
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            ),
        ),
    ]
    num_images_per_prompt: Annotated[
        int,
        Field(default=1, description="Number of images to generate per prompt."),
    ]


RESPONSES = {
    status.HTTP_200_OK: {
        "content": {
            "application/json": {
                "schema": {
                    "x-speakeasy-name-override": "data",
                }
            }
        },
    },
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


@router.post(
    "/text-to-image",
    response_model=ImageResponse,
    responses=RESPONSES,
    description="Generate images from text prompts.",
    operation_id="genTextToImage",
    summary="Text To Image",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "textToImage"},
)
@router.post(
    "/text-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_image(
    params: TextToImageParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    # Ensure required parameters are non-empty.
    # TODO: Remove if go-livepeer validation is fixed. Was disabled due to optional
    # params issue.
    if not params.prompt:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Prompt must be provided."),
        )

    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token."),
            )

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}."
            ),
        )

    seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(params.num_images_per_prompt)]

    # TODO: Process one image at a time to avoid CUDA OEM errors. Can be removed again
    # once LIV-243 and LIV-379 are resolved.
    images = []
    has_nsfw_concept = []
    params.num_images_per_prompt = 1
    for seed in seeds:
        params.seed = seed
        kwargs = {k: v for k, v in params.model_dump().items() if k != "model_id"}
        try:
            imgs, nsfw_check = await asyncio.to_thread(pipeline, **kwargs)
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                # TODO: Investigate why not all VRAM memory is cleared.
                torch.cuda.empty_cache()
            logger.error(f"TextToImage pipeline error: {e}")
            return handle_pipeline_exception(
                e,
                default_error_message="Text-to-image pipeline error.",
                custom_error_config=PIPELINE_ERROR_CONFIG,
            )
        images.extend(imgs)
        has_nsfw_concept.extend(nsfw_check)

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}

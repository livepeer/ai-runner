import logging
import os
import random
from typing import Annotated, Dict, Tuple, Union

import torch
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    VideoResponse,
    handle_pipeline_exception,
    http_error,
    image_to_data_url,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input or output video resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}


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


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post(
    "/image-to-video",
    response_model=VideoResponse,
    responses=RESPONSES,
    description="Generate a video from a provided image.",
    operation_id="genImageToVideo",
    summary="Image To Video",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "imageToVideo"},
)
@router.post(
    "/image-to-video/",
    response_model=VideoResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_video(
    image: Annotated[
        UploadFile,
        File(description="Uploaded image to generate a video from."),
    ],
    prompt: Annotated[
        str,
        Form(description="Text prompt(s) to guide video generation for prompt accepting models.")
    ] = "",
    negative_prompt: Annotated[
        str,
        Form(
            description=(
                "Text prompt(s) to guide what to exclude from video generation for prompt accepting models. "
                "Ignored if guidance_scale < 1."
            )
        ),
    ] = "",
    model_id: Annotated[
        str, Form(description="Hugging Face model ID used for video generation.")
    ] = "",
    height: Annotated[
        int, Form(description="The height in pixels of the generated video.")
    ] = 576,
    width: Annotated[
        int, Form(description="The width in pixels of the generated video.")
    ] = 1024,
    fps: Annotated[
        int, Form(description="The frames per second of the generated video.")
    ] = 6,
    motion_bucket_id: Annotated[
        int,
        Form(
            description=(
                "Used for conditioning the amount of motion for the generation. The "
                "higher the number the more motion will be in the video."
            )
        ),
    ] = 127,
    noise_aug_strength: Annotated[
        float,
        Form(
            description=(
                "Amount of noise added to the conditioning image. Higher values reduce "
                "resemblance to the conditioning image and increase motion."
            )
        ),
    ] = 0.02,
    safety_check: Annotated[
        bool,
        Form(
            description=(
                "Perform a safety check to estimate if generated images could be "
                "offensive or harmful."
            )
        ),
    ] = True,
    seed: Annotated[int, Form(description="Seed for random number generation.")] = None,
    num_inference_steps: Annotated[
        int,
        Form(
            description=(
                "Number of denoising steps. More steps usually lead to higher quality "
                "images but slower inference. Modulated by strength."
            )
        ),
    ] = 25,  # NOTE: Hardcoded due to varying pipeline values.
    num_frames: Annotated[
        int, Form(description="The number of video frames to generate.")
    ] = 25, # NOTE: Added `25` as default value to consider for `stable-video-diffusion-img2vid-xt` model having smaller default value than LTX-V in its pipeline.
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token."),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}."
            ),
        )

    if height % 8 != 0 or width % 8 != 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"`height` and `width` have to be divisible by 8 but are {height} and "
                f"{width}."
            ),
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    try:
        batch_frames, has_nsfw_concept = pipeline(
            image=Image.open(image.file).convert("RGB"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
            safety_check=safety_check,
            seed=seed,
        )
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # TODO: Investigate why not all VRAM memory is cleared.
            torch.cuda.empty_cache()
        logger.error(f"ImageToVideo pipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Image-to-video pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    output_frames = []
    for frames in batch_frames:
        output_frames.append(
            [
                {
                    "url": image_to_data_url(frame),
                    "seed": seed,
                    "nsfw": has_nsfw_concept[0],
                }
                for frame in frames
            ]
        )

    return {"frames": output_frames}

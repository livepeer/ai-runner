import logging
import os
import random
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.util import HTTPError, ImageResponse, http_error, image_to_data_url

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)


RESPONSES = {
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}


# TODO: Make model_id and other None properties optional once Go codegen tool supports
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-image", response_model=ImageResponse, responses=RESPONSES)
@router.post(
    "/image-to-image/",
    response_model=ImageResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def image_to_image(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    strength: Annotated[float, Form()] = 0.8,
    guidance_scale: Annotated[float, Form()] = 7.5,
    image_guidance_scale: Annotated[float, Form()] = 1.5,
    negative_prompt: Annotated[str, Form()] = "",
    safety_check: Annotated[bool, Form()] = True,
    seed: Annotated[int, Form()] = None,
    num_inference_steps: Annotated[
        int, Form()
    ] = 100,  # NOTE: Hardcoded due to varying pipeline values.
    num_images_per_prompt: Annotated[int, Form()] = 1,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    seeds = [seed + i for i in range(num_images_per_prompt)]

    image = Image.open(image.file).convert("RGB")

    # TODO: Process one image at a time to avoid CUDA OEM errors. Can be removed again
    # once LIV-243 and LIV-379 are resolved.
    images = []
    has_nsfw_concept = []
    for seed in seeds:
        try:
            imgs, nsfw_checks = pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                negative_prompt=negative_prompt,
                safety_check=safety_check,
                seed=seed,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps,
            )
            images.extend(imgs)
            has_nsfw_concept.extend(nsfw_checks)
        except Exception as e:
            logger.error(f"ImageToImagePipeline error: {e}")
            logger.exception(e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=http_error("ImageToImagePipeline error"),
            )

    # TODO: Return None once Go codegen tool supports optional properties
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    output_images = [
        {"url": image_to_data_url(img), "seed": sd, "nsfw": nsfw or False}
        for img, sd, nsfw in zip(images, seeds, has_nsfw_concept)
    ]

    return {"images": output_images}

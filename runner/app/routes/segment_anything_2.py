import logging
import os
from typing import Annotated, Dict, Tuple, Union

import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, ImageFile

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    HTTPError,
    MasksResponse,
    handle_pipeline_exception,
    http_error,
    json_str_to_np_array,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)


# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing input image resolution.",
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
# OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373.
@router.post(
    "/segment-anything-2",
    response_model=MasksResponse,
    responses=RESPONSES,
    description="Segment objects in an image.",
    operation_id="genSegmentAnything2",
    summary="Segment Anything 2",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "segmentAnything2"},
)
@router.post(
    "/segment-anything-2/",
    response_model=MasksResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
def segment_anything_2(
    image: Annotated[
        UploadFile, File(description="Image to segment.", media_type="image/*")
    ],
    model_id: Annotated[
        str, Form(description="Hugging Face model ID used for image generation.")
    ] = "",
    point_coords: Annotated[
        str,
        Form(
            description=(
                "Nx2 array of point prompts to the model, where each point is in (X,Y) "
                "in pixels."
            )
        ),
    ] = None,
    point_labels: Annotated[
        str,
        Form(
            description=(
                "Labels for the point prompts, where 1 indicates a foreground point "
                "and 0 indicates a background point."
            )
        ),
    ] = None,
    box: Annotated[
        str,
        Form(
            description=(
                "A length 4 array given as a box prompt to the model, in XYXY format."
            )
        ),
    ] = None,
    mask_input: Annotated[
        str,
        Form(
            description=(
                "A low-resolution mask input to the model, typically from a previous "
                "prediction iteration, with the form 1xHxW (H=W=256 for SAM)."
            )
        ),
    ] = None,
    multimask_output: Annotated[
        bool,
        Form(
            description=(
                "If true, the model will return three masks for ambiguous input "
                "prompts, often producing better masks than a single prediction."
            )
        ),
    ] = True,
    return_logits: Annotated[
        bool,
        Form(
            description=(
                "If true, returns un-thresholded mask logits instead of a binary mask."
            )
        ),
    ] = True,
    normalize_coords: Annotated[
        bool,
        Form(
            description=(
                "If true, the point coordinates will be normalized to the range [0,1], "
                "with point_coords expected to be with respect to image dimensions."
            )
        ),
    ] = True,
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

    try:
        point_coords = json_str_to_np_array(point_coords, var_name="point_coords")
        point_labels = json_str_to_np_array(point_labels, var_name="point_labels")
        box = json_str_to_np_array(box, var_name="box")
        mask_input = json_str_to_np_array(mask_input, var_name="mask_input")
    except ValueError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(str(e)),
        )

    try:
        image = Image.open(image.file).convert("RGB")
        masks, scores, low_res_mask_logits = pipeline(
            image,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # TODO: Investigate why not all VRAM memory is cleared.
            torch.cuda.empty_cache()
        logger.error(f"SegmentAnything2 pipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Segment-anything-2 pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    # Return masks sorted by descending score as string.
    sorted_ind = np.argsort(scores)[::-1]
    return {
        "masks": str(masks[sorted_ind].tolist()),
        "scores": str(scores[sorted_ind].tolist()),
        "logits": str(low_res_mask_logits[sorted_ind].tolist()),
    }

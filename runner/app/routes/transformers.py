import logging
import torch
import os
from typing import Union, Annotated, Dict, Tuple, Any
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import APIRouter, status, Depends
from pydantic import BaseModel, Field, HttpUrl
from transformers import pipeline
from app.pipelines.utils import get_torch_device

from app.routes.utils import http_error, handle_pipeline_exception, HTTPError

router = APIRouter()

logger = logging.getLogger(__name__)

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Error strings.
    "Unknown task string": (
        "",
        status.HTTP_400_BAD_REQUEST,
    ),
    "unexpected keyword argument": (
        "Unexpected keyword argument provided.",
        status.HTTP_400_BAD_REQUEST,
    ),
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing output image resolution.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    ),
}


class InferenceRequest(BaseModel):
    # TODO: Make params optional once Go codegen tool supports OAPI 3.1
    # https://github.com/deepmap/oapi-codegen/issues/373
    task: Annotated[
        str,
        Field(
            description=(
                "The transformer task to perform. E.g. 'automatic-speech-recognition'."
            ),
        ),
    ]
    model_name: Annotated[
        str,
        Field(
            description=(
                "The transformer model to use for the task. E.g. 'openai/whisper-base'."
            ),
        ),
    ]
    input: Annotated[
        Union[str, HttpUrl],
        Field(
            description=(
                "The input data to be transformed. Can be string or an url to a file."
            ),
        ),
    ]
    pipeline_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the transformer pipeline during inference. E.g. {'return_timestamps': True, 'max_length': 50}.",
    )


class InferenceResponse(BaseModel):
    """Response model for transformer inference."""

    output: Any = Field(
        ..., description="The output data transformed by the transformer pipeline."
    )


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
    "/transformers",
    response_model=InferenceResponse,
    responses=RESPONSES,
    description="Perform inference using a Hugging Face transformer model.",
    operation_id="genTransformers",
    summary="Transformers",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "transformers"},
)
@router.post("/transformers/", responses=RESPONSES, include_in_schema=False)
async def transformers(
    request: InferenceRequest,
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

    if not request.task and not request.model_name:
        raise JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Either 'task' or 'model_name' must be provided."),
        )
    if not request.input:
        raise JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("'input' field is required."),
        )

    torch_device = get_torch_device()

    # Initialize the pipeline with the specified task and model ID.
    pipeline_kwargs = {}
    if request.task:
        pipeline_kwargs["task"] = request.task
    if request.model_name:
        pipeline_kwargs["model"] = request.model_name
    try:
        pipe = pipeline(device=torch_device, **pipeline_kwargs)
    except Exception as e:
        return handle_pipeline_exception(
            e,
            default_error_message=f"Pipeline initialization error: {e}.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    # Perform inference using the pipeline.
    try:
        out = pipe(request.input, **request.pipeline_params)
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # TODO: Investigate why not all VRAM memory is cleared.
            torch.cuda.empty_cache()
        logger.error(f"TransformersPipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="transformers pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )

    return {"output": out}

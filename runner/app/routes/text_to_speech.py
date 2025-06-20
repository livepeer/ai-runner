import base64
import logging
import os
import time
from typing import Annotated, Dict, Tuple, Union, Optional
import torch
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    AudioResponse,
    HTTPError,
    audio_to_data_url,
    handle_pipeline_exception,
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# ---------------- Validation limits ----------------
MAX_TEXT_LEN = 10000  # maximum characters for text
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB

# Pipeline specific error handling configuration.
PIPELINE_ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "OutOfMemoryError": (
        "Out of memory error. Try reducing text input length.",
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}

class TextToSpeechParams(BaseModel):

    # TODO: Make model_id and other None properties optional once Go codegen tool
    # supports OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    model_id: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Optional Hugging Face model ID for text-to-speech generation. If omitted, the pipelines configured model is used.",
        ),
    ]
    text: Annotated[
        str,
        Field(
            default=(
                "When it was all over, the remaing animals, except for the pigs and dogs, "
                "crept away in a body. They were shaken and miserable. They did not know "
                "which was more shocking - the treachery of the animals who had leagued "
                "themselves with Snowball, or the cruel retribution they had just witnessed."
            ),
            description="Text input for speech generation.",
        ),
    ]
    audio_prompt_base64: Annotated[
        bytes | None,
        Field(
            default=None,
            description=(
                "Optional base64-encoded audio data for voice cloning reference. Provide as base64-encoded string; it will be decoded server-side. Must be a valid audio file format like WAV or MP3."
            ),
        ),
    ]

    @validator("text")
    def validate_text_length(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > MAX_TEXT_LEN:
            raise ValueError(f"text exceeds {MAX_TEXT_LEN} characters")
        return v
        
    @validator("audio_prompt_base64", pre=True)
    def validate_and_decode_audio_prompt(cls, v):
        """Decode base64 audio once during validation and enforce size limits."""
        if v is None:
            return None
        try:
            # Accept already-bytes input for flexibility.
            if isinstance(v, (bytes, bytearray)):
                decoded = bytes(v)
            else:
                decoded = base64.b64decode(v)
            if len(decoded) > MAX_AUDIO_BYTES:
                raise ValueError(
                    f"decoded audio data exceeds {MAX_AUDIO_BYTES / (1024 * 1024):.1f} MB"
                )
            return decoded
        except Exception as e:
            raise ValueError(f"invalid base64 audio data: {e}") from e


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
    "/text-to-speech",
    response_model=AudioResponse,
    responses=RESPONSES,
    description=(
        "Generate a text-to-speech audio file based on the provided text input."
        "Optionally include base64-encoded audio for voice cloning."
    ),
    operation_id="genTextToSpeech",
    summary="Text To Speech",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "textToSpeech"},
)
@router.post(
    "/text-to-speech/",
    response_model=AudioResponse,
    responses=RESPONSES,
    include_in_schema=False,
)
async def text_to_speech(
    params: TextToSpeechParams,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    # Ensure required parameters are non-empty.
    # TODO: Remove if go-livepeer validation is fixed. Was disabled due to optional
    # params issue.
    if not params.text:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error("Text input must be provided."),
        )

    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token."),
            )



    # If a model_id is supplied and differs from the pipeline’s current model, log a warning.
    if params.model_id and params.model_id != pipeline.model_id:
        logger.warning(
            "Requested model_id %s differs from pipeline model_id %s — proceeding with current pipeline.",
            params.model_id,
            pipeline.model_id,
        )


    try:
        start_time = time.time()
        output = pipeline(params)
        end_time = time.time()
        logger.info(f"TextToSpeechPipeline took {end_time - start_time} seconds.")
    except Exception as e:
        if isinstance(e, torch.cuda.OutOfMemoryError):
            # TODO: Investigate why not all VRAM memory is cleared.
            torch.cuda.empty_cache()
        logger.error(f"TextToSpeechPipeline error: {e}")
        return handle_pipeline_exception(
            e,
            default_error_message="Text-to-speech pipeline error.",
            custom_error_config=PIPELINE_ERROR_CONFIG,
        )


    return {"audio": {"url": audio_to_data_url(output)}}

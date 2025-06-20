import logging
import os
import time
from typing import Annotated, Dict, Tuple, Union
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import (
    AudioResponse,
    HTTPError,
    audio_to_data_url,
    file_exceeds_max_size,
    handle_pipeline_exception,
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# ---------------- Validation limits ----------------
MAX_TEXT_LEN = 10000  # maximum characters for text
MAX_DESC_LEN = 1000  # maximum characters for description
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
        str,
        Field(
            default="ResembleAI/chatterbox",
            description="Hugging Face model ID used for text to speech generation.",
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
    description: Annotated[
        str | None,
        Field(
            default=(
                "A male speaker delivers a slightly expressive and animated speech "
                "with a moderate speed and pitch."
            ),
            description=("Description of speaker to steer text to speech generation."),
        ),
    ]
    audio_prompt_path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional path or URL to a reference audio clip for voice cloning (only used when model_id refers to Chatterbox)."
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
        
    @validator("description")
    def validate_description_length(cls, v):
        if len(v) > MAX_DESC_LEN:
            raise ValueError(f"description exceeds {MAX_DESC_LEN} characters")
        return v


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


def _multipart_params(
    model_id: str = Form(""),
    text: str = Form(""),
    description: str = Form(
        "A male speaker delivers a slightly expressive and animated speech with a moderate speed and pitch."
    ),
) -> "TextToSpeechParams":
    return TextToSpeechParams(
        model_id=model_id,
        text=text,
        description=description,
    )


@router.post(
    "/text-to-speech",
    response_model=AudioResponse,
    responses=RESPONSES,
    description=(
        "Generate a text-to-speech audio file based on the provided text input and "
        "speaker description."
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
    params: TextToSpeechParams = Depends(_multipart_params),
    audio_prompt: UploadFile | None = File(None, description="Optional reference audio file for Chatterbox voice cloning"),
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

    # Check uploaded file size early
    if audio_prompt is not None and file_exceeds_max_size(audio_prompt, MAX_AUDIO_BYTES):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=http_error("audio_prompt too large; 10 MB max"),
        )

    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{params.model_id}"
            ),
        )

    # Handle uploaded audio prompt.
    temp_path = None
    if audio_prompt is not None:
        suffix = Path(audio_prompt.filename or "audio").suffix or ".wav"
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio_prompt.read())
            temp_path = tmp.name
        params.audio_prompt_path = temp_path  # override

    try:
        start = time.time()
        out = pipeline(params)
        logger.info(f"TextToSpeechPipeline took {time.time() - start} seconds.")
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
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return {"audio": {"url": audio_to_data_url(out)}}

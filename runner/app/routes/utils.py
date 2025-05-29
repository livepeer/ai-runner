import base64
import io
import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field


class MediaURL(BaseModel):
    """A URL from which media can be accessed."""

    url: str = Field(..., description="The URL where the media can be accessed.")


class Media(MediaURL):
    """A media object containing information about the generated media."""

    seed: int = Field(..., description="The seed used to generate the media.")
    # TODO: Make nsfw property optional once Go codegen tool supports
    # OAPI 3.1 https://github.com/deepmap/oapi-codegen/issues/373
    nsfw: bool = Field(..., description="Whether the media was flagged as NSFW.")


class ImageResponse(BaseModel):
    """Response model for image generation."""

    images: List[Media] = Field(..., description="The generated images.")


class VideoResponse(BaseModel):
    """Response model for video generation."""

    frames: List[List[Media]] = Field(..., description="The generated video frames.")


class AudioResponse(BaseModel):
    """Response model for audio generation."""

    audio: MediaURL = Field(..., description="The generated audio.")


class MasksResponse(BaseModel):
    """Response model for object segmentation."""

    masks: str = Field(..., description="The generated masks.")
    scores: str = Field(
        ..., description="The model's confidence scores for each generated mask."
    )
    logits: str = Field(
        ..., description="The raw, unnormalized predictions (logits) for the masks."
    )


class Chunk(BaseModel):
    """A chunk of text with a timestamp."""

    timestamp: Tuple = Field(..., description="The timestamp of the chunk.")
    text: str = Field(..., description="The text of the chunk.")


class TextResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="The generated text.")
    chunks: List[Chunk] = Field(..., description="The generated text chunks.")


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMBaseChoice(BaseModel):
    index: int
    finish_reason: str = "" # Needs OpenAPI 3.1 support to make optional


class LLMTokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class LLMChoice(LLMBaseChoice):
    delta: LLMMessage = None
    message: LLMMessage = None

class LLMResponse(BaseModel):
    id: str
    model: str
    created: int
    usage: LLMTokenUsage
    choices: List[LLMChoice]

class LLMRequest(BaseModel):
    messages: List[LLMMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    top_k: int = -1
    stream: bool = False


class ImageToTextResponse(BaseModel):
    """Response model for text generation."""

    text: str = Field(..., description="The generated text.")


class LiveVideoToVideoResponse(BaseModel):
    """Response model for live video-to-video generation."""

    subscribe_url: str = Field(
        ..., description="Source URL of the incoming stream to subscribe to"
    )
    publish_url: str = Field(
        ..., description="Destination URL of the outgoing stream to publish to"
    )
    control_url: str = Field(
        default='',
        description="URL for updating the live video-to-video generation",
    )
    events_url: str = Field(
        default='',
        description="URL for subscribing to events for pipeline status and logs",
    )
    request_id: str = Field(
        default='',
        description="The ID generated for this request",
    )
    manifest_id: str = Field(
        default='',
        description="Orchestrator manifest ID for this request",
    )


class APIError(BaseModel):
    """API error response model."""

    msg: str = Field(..., description="The error message.")


class HTTPError(BaseModel):
    """HTTP error response model."""

    detail: APIError = Field(..., description="Detailed error information.")


def http_error(msg: str) -> HTTPError:
    """Create an HTTP error response with the specified message.

    Args:
        msg: The error message.

    Returns:
        The HTTP error response.
    """
    return {"detail": {"msg": msg}}


def image_to_base64(img: Image, format: str = "png") -> str:
    """Convert an image to a base64 string.

    Args:
        img: The image to convert.
        format: The image format to use. Defaults to "png".

    Returns:
        The base64-encoded image.
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_data_url(img: Image, format: str = "png") -> str:
    """Convert an image to a data URL.

    Args:
        img: The image to convert.
        format: The image format to use. Defaults to "png".

    Returns:
        The data URL for the image.
    """
    return "data:image/png;base64," + image_to_base64(img, format=format)


def audio_to_data_url(buffer: io.BytesIO, format: str = "wav") -> str:
    """Convert an audio buffer to a data URL.

    Args:
        buffer: The audio buffer to convert.
        format: The audio format to use. Defaults to "wav".

    Returns:
        The data URL for the audio.
    """
    base64_audio = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:audio/{format};base64,{base64_audio}"


def file_exceeds_max_size(
    input_file: UploadFile, max_size: int = 10 * 1024 * 1024
) -> bool:
    """Checks if the uploaded file exceeds the specified maximum size.

    Args:
        input_file: The uploaded file to check.
        max_size: The maximum allowed file size in bytes. Defaults to 10 MB.

    Returns:
        True if the file exceeds the maximum size, False otherwise.
    """
    try:
        if input_file.file:
            # Get size by moving the cursor to the end of the file and back.
            input_file.file.seek(0, os.SEEK_END)
            file_size = input_file.file.tell()
            input_file.file.seek(0)
            return file_size > max_size
    except Exception as e:
        print(f"Error checking file size: {e}")
    return False


def json_str_to_np_array(
    data: Optional[str], var_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """Converts a JSON string to a NumPy array.

    Args:
        data: The JSON string to convert.
        var_name: The name of the variable being converted. Used in error messages.

    Returns:
        The NumPy array if the conversion is successful, None otherwise.

    Raises:
        ValueError: If an error occurs during JSON parsing.
    """
    if data:
        try:
            array = np.array(json.loads(data))
            return array
        except json.JSONDecodeError as e:
            error_message = "Error parsing JSON"
            if var_name:
                error_message += f" for {var_name}"
            error_message += f": {e}"
            raise ValueError(error_message)
    return None


# Global error handling configuration.
# NOTE: "" for default message, None for exception message.
ERROR_CONFIG: Dict[str, Tuple[Union[str, None], int]] = {
    # Specific error types.
    "LoraLoadingError": (None, status.HTTP_400_BAD_REQUEST),
    "InferenceError": (None, status.HTTP_400_BAD_REQUEST),
    "ValueError": ("", status.HTTP_400_BAD_REQUEST),
    "OutOfMemoryError": ("GPU out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
    # General error patterns.
    "out of memory": ("Out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
    "CUDA out of memory": ("GPU out of memory.", status.HTTP_500_INTERNAL_SERVER_ERROR),
}


def handle_pipeline_exception(
    e: object,
    default_error_message: Union[str, Dict[str, object], None] = "Pipeline error.",
    default_status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    custom_error_config: Dict[str, Tuple[str | None, int]] | None = None,
) -> JSONResponse:
    """Handles pipeline exceptions by returning a JSON response with the appropriate
    error message and status code.

    Args:
        e(int): The exception to handle. Can be any type of object.
        default_error_message: The default error message or content dictionary. Default
            will be used if no specific error type ismatched.
        default_status_code: The default status code to use if no specific error type is
            matched. Defaults to HTTP_500_INTERNAL_SERVER_ERROR.
        custom_error_config: Custom error configuration to override the application
            error configuration.

    Returns:
        The JSON response with appropriate status code and error message.
    """
    error_config = ERROR_CONFIG.copy()
    if custom_error_config:
        error_config.update(custom_error_config)

    error_message = default_error_message
    status_code = default_status_code

    error_type = type(e).__name__
    if error_type in error_config:
        error_message, status_code = error_config[error_type]
    else:
        for error_pattern, (message, code) in error_config.items():
            if error_pattern.lower() in str(e).lower():
                status_code = code
                error_message = message
                break

    if error_message is None:
        error_message = f"{e}."
    elif error_message == "":
        error_message = default_error_message

    content = (
        http_error(error_message) if isinstance(error_message, str) else error_message
    )

    return JSONResponse(
        status_code=status_code,
        content=content,
    )


def parse_key_from_metadata(
    metadata: str, key: str, expected_type: type
) -> Union[Optional[Union[str, int, float, bool]]]:
    """Parse a specific key from the metadata JSON string.

    Args:
        metadata: The metadata JSON string.
        key: The key to parse from the metadata.
        expected_type: The expected type of the key's value.

     Returns:
        The value of the key if it exists and is of the expected type, otherwise None.

    Raises:
        ValueError: If the metadata is not valid JSON.
        TypeError: If the value is not of the expected type.
    """
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    value = metadata_dict.get(key)
    if value is not None:
        if isinstance(value, expected_type):
            return value
        try:
            return expected_type(value)
        except (ValueError, TypeError):
            raise TypeError(
                f"Invalid {key} value. Must be of type {expected_type.__name__}."
            )
    return None


def get_media_duration_ffmpeg(bytes: bytes) -> float:
    """Gets the duration of the media using ffprobe.

    Args:
        bytes: The media file as bytes.

    Returns:
        The duration of the media in seconds.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(bytes)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                temp_file_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
    except Exception as e:
        raise Exception(f"Failed to get duration with ffmpeg: {e}")
    finally:
        os.remove(temp_file_path)

    return duration

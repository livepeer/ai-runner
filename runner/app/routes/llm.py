import logging
import os
from typing import Union
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline
from app.routes.utils import HTTPError, LLMRequest, LLMChoice, LLMMessage, LLMResponse, http_error
import json

router = APIRouter()
logger = logging.getLogger(__name__)

RESPONSES = {
    status.HTTP_200_OK: {"model": LLMResponse},
    status.HTTP_400_BAD_REQUEST: {"model": HTTPError},
    status.HTTP_401_UNAUTHORIZED: {"model": HTTPError},
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPError},
}

@router.post(
    "/llm",
    response_model=LLMResponse
,
    responses=RESPONSES,
    operation_id="genLLM",
    description="Generate text using a language model.",
    summary="LLM",
    tags=["generate"],
    openapi_extra={"x-speakeasy-name-override": "llm"},
)
@router.post("/llm/", response_model=LLMResponse
, responses=RESPONSES, include_in_schema=False)
async def llm(
    request: LLMRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
) -> Union[LLMResponse, JSONResponse, StreamingResponse]:
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if request.model != "" and request.model != pipeline.model_id:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with {request.model}"
            ),
        )

    try:
        generator = pipeline(
            messages=[msg.dict() for msg in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k
        )

        if request.stream:
            return StreamingResponse(
                stream_generator(generator),
                media_type="text/event-stream"
            )
        else:
            full_response = ""
            last_chunk = None
            async for chunk in generator:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                last_chunk = chunk

            if last_chunk:
                # Return the final response with accumulated text
                return LLMResponse(
                    choices=[
                        LLMChoice(
                            message=LLMMessage(
                                role="assistant",
                                content=full_response
                            ),
                            index=0,
                            finish_reason="stop"
                        )
                    ],
                    usage=last_chunk.usage,
                    id=last_chunk.id,
                    model=last_chunk.model,
                    created=last_chunk.created
                )
            else:
                raise ValueError("No response generated")

    except Exception as e:
        logger.error(f"LLM processing error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error(
                "Internal server error during LLM processing."
            )
        )


async def stream_generator(generator):
    try:
        async for chunk in generator:
            if isinstance(chunk, LLMResponse):
                if len(chunk.choices) > 0:
                    # Regular streaming chunk or final chunk
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    if chunk.choices[0].finish_reason == "stop":
                        break
        # Signal end of stream
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

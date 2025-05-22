import logging

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.dependencies import get_pipeline
from app.pipelines.base import Pipeline, HealthCheck
from app.routes.utils import (
    http_error,
)

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/health", operation_id="health", response_model=HealthCheck)
@router.get("/health/", response_model=HealthCheck, include_in_schema=False)
def health(pipeline: Pipeline = Depends(get_pipeline)) -> HealthCheck | JSONResponse:
    try:
        return pipeline.get_health()
    except Exception as e:
        logger.error(f"Error retrieving pipeline status: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=http_error("Failed to retrieve pipeline status."),
        )

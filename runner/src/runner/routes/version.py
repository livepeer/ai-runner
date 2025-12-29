import os

from fastapi import APIRouter, Request
from runner.pipelines.base import Version

router = APIRouter()

@router.get("/version", operation_id="version", response_model=Version)
@router.get("/version/", response_model=Version, include_in_schema=False)
def version(request: Request) -> Version:
    pipeline = request.app.pipeline
    return Version(
        pipeline=os.environ.get("PIPELINE", pipeline.name),
        model_id=os.environ.get("MODEL_ID", pipeline.model_id),
        version=os.environ.get("VERSION", "undefined"),
    )

import os

from fastapi import APIRouter
from app.pipelines.base import VersionCheck

router = APIRouter()

@router.get("/version", operation_id="version", response_model=VersionCheck)
@router.get("/version/", response_model=VersionCheck, include_in_schema=False)
def version() -> VersionCheck:
    return VersionCheck(
        version=os.environ["VERSION"],
    )

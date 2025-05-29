import asyncio
import os
from typing import Dict

from app.utils.hardware import (
    GPUComputeInfo,
    GPUUtilizationInfo
)
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HardwareInformation(BaseModel):
    """Response model for GPU information."""

    pipeline: str
    model_id: str
    gpu_info: Dict[int, GPUComputeInfo]


class HardwareStats(BaseModel):
    """Response model for real-time GPU statistics."""

    pipeline: str
    model_id: str
    gpu_stats: Dict[int, GPUUtilizationInfo]


@router.get(
    "/hardware/info",
    operation_id="hardware_info",
    response_model=HardwareInformation,
)
@router.get(
    "/hardware/info/",
    response_model=HardwareInformation,
    include_in_schema=False,
)
async def hardware_info(request: Request):
    gpu_info = await asyncio.to_thread(request.app.hardware_info_service.get_gpu_compute_info)
    return HardwareInformation(
        pipeline=os.environ["PIPELINE"],
        model_id=os.environ["MODEL_ID"],
        gpu_info=gpu_info,
    )


@router.get(
    "/hardware/stats",
    operation_id="hardware_stats",
    response_model=HardwareStats,
)
@router.get(
    "/hardware/stats/",
    response_model=HardwareStats,
    include_in_schema=False,
)
async def hardware_stats(request: Request):
    gpu_stats = await asyncio.to_thread(request.app.hardware_info_service.get_gpu_utilization_stats)
    return HardwareStats(
        pipeline=os.environ["PIPELINE"],
        model_id=os.environ["MODEL_ID"],
        gpu_stats=gpu_stats,
    )

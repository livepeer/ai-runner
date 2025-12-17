"""Contains utility functions for hardware information."""

from typing import Dict
from pydantic import BaseModel
import logging
import pynvml
import atexit

logger = logging.getLogger(__name__)


class GPUBaseInfo(BaseModel):
    """Model for general GPU information."""

    id: str
    name: str
    memory_total: int
    memory_free: int


class GPUComputeInfo(GPUBaseInfo):
    """Model for detailed GPU compute information."""

    major: int
    minor: int


class GPUUtilizationInfo(GPUBaseInfo):
    """Model for GPU utilization statistics."""

    utilization_compute: int
    utilization_memory: int


class GPUInfo(GPUComputeInfo, GPUUtilizationInfo):
    """Model for full GPU device information."""

    pass


class HardwareInfo:
    """Class used to retrieve hardware information about the host machine."""

    def __init__(self):
        """Initialize the HardwareInfo class and hardware info retrieval services."""
        self._initialized = False
        self._initialize_nvml()
        atexit.register(self._shutdown_nvml)

    def _initialize_nvml(self) -> None:
        """Initialize NVML."""
        if not self._initialized:
            try:
                pynvml.nvmlInit()
                self._initialized = True
                logger.info("NVML initialized successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Failed to initialize NVML: {e}")

    def _shutdown_nvml(self) -> None:
        """Shutdown NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
                logger.info("NVML shutdown successfully.")
            except pynvml.NVMLError as e:
                logger.error(f"Failed to shutdown NVML: {e}")

    def get_cuda_info(self) -> Dict[int, GPUInfo]:
        """Retrieve CUDA device information.

        Returns:
            A dictionary mapping GPU device IDs to their information.
        """
        devices = {}
        if not self._initialized:
            logger.warning(
                "NVML is not initialized. Cannot retrieve CUDA device information."
            )
            return devices

        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                utilization_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                devices[i] = GPUInfo(
                    id=uuid,
                    name=name,
                    memory_total=memory_info.total,
                    memory_free=memory_info.free,
                    major=major,
                    minor=minor,
                    utilization_compute=utilization_rates.gpu,
                    utilization_memory=utilization_rates.memory,
                )
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to retrieve CUDA device information: {e}")
        return devices

    def get_gpu_compute_info(self) -> Dict[int, GPUComputeInfo]:
        """Get detailed GPU compute information.

        Returns:
            A dictionary mapping GPU device IDs to their compute information.
        """
        basic_info = self.get_cuda_info()
        return {
            i: GPUComputeInfo(
                id=info.id,
                name=info.name,
                memory_total=info.memory_total,
                memory_free=info.memory_free,
                major=info.major,
                minor=info.minor,
            )
            for i, info in basic_info.items()
        }

    def log_gpu_compute_info(self):
        """Log detailed GPU compute information."""
        devices = self.get_gpu_compute_info()
        if devices:
            logger.info("CUDA devices available:")
            for device_id, device_info in devices.items():
                logger.info(f"Device {device_id}: {device_info}")
        else:
            logger.info("No CUDA devices available.")

    def get_gpu_utilization_stats(self) -> Dict[int, GPUUtilizationInfo]:
        """Get GPU utilization statistics.

        Returns:
            A dictionary mapping GPU device IDs to their utilization statistics.
        """
        basic_info = self.get_cuda_info()
        return {
            i: GPUUtilizationInfo(
                id=info.id,
                name=info.name,
                memory_total=info.memory_total,
                memory_free=info.memory_free,
                utilization_compute=info.utilization_compute,
                utilization_memory=info.utilization_memory,
            )
            for i, info in basic_info.items()
        }

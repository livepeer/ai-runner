import pytest
import pynvml
from runner.utils.hardware import HardwareInfo, GPUInfo, GPUComputeInfo, GPUUtilizationInfo

class TestHardwareInfo:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        self.mocker = mocker

        self.mock_pynvml = self.mocker.patch("runner.utils.hardware.pynvml")
        self.mock_pynvml.NVMLError = pynvml.NVMLError
        self.mock_pynvml.nvmlInit.return_value = None
        self.mock_pynvml.nvmlShutdown.return_value = None
        self.mock_pynvml.nvmlDeviceGetCount.return_value = self.mocker.MagicMock(value=1)
        self.mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "GPU-Handle"
        self.mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-UUID"
        self.mock_pynvml.nvmlDeviceGetName.return_value = "GPU-Name"
        self.mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = self.mocker.MagicMock(total=1024, free=512)
        self.mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (7, 5)
        self.mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = self.mocker.MagicMock(gpu=50, memory=30)

        self.hardware_info = HardwareInfo()

        yield

        self.hardware_info._initialized = False

    def test_init_success(self):
        assert self.hardware_info._initialized is True, "HardwareInfo should be initialized"

    def test_init_failure(self):
        self.mock_pynvml.nvmlInit.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND)
        hardware_info = HardwareInfo()
        assert hardware_info._initialized is False, "HardwareInfo should not be initialized"


    def test_get_cuda_info_success(self):
        devices = self.hardware_info.get_cuda_info()
        assert len(devices) == 1, "Should retrieve one CUDA device"
        device = devices[0]
        expected_device = GPUInfo(
            id="GPU-UUID",
            name="GPU-Name",
            memory_total=1024,
            memory_free=512,
            major=7,
            minor=5,
            utilization_compute=50,
            utilization_memory=30,
        )
        assert device == expected_device, "CUDA device information should match the expected values"

    def test_get_cuda_info_nvml_not_initialized(self):
        self.hardware_info._initialized = False
        devices = self.hardware_info.get_cuda_info()
        assert len(devices) == 0, "Should not retrieve any CUDA devices when NVML is not initialized"

    def test_get_cuda_info_nvml_error(self):
        self.mock_pynvml.nvmlDeviceGetCount.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        devices = self.hardware_info.get_cuda_info()
        assert len(devices) == 0, "Should not retrieve any CUDA devices when NVML encounters an error"

    def test_get_gpu_compute_info_return_type(self):
        compute_info = self.hardware_info.get_gpu_compute_info()
        assert isinstance(compute_info, dict), "Return type should be a dictionary"
        for key, value in compute_info.items():
            assert isinstance(key, int), "Dictionary keys should be integers"
            assert isinstance(value, GPUComputeInfo), "Dictionary values should be GPUComputeInfo instances"

    def test_get_gpu_utilization_stats_return_type(self):
        utilization_stats = self.hardware_info.get_gpu_utilization_stats()
        assert isinstance(utilization_stats, dict), "Return type should be a dictionary"
        for key, value in utilization_stats.items():
            assert isinstance(key, int), "Dictionary keys should be integers"
            assert isinstance(value, GPUUtilizationInfo), "Dictionary values should be GPUUtilizationInfo instances"

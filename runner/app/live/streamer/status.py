import time
import hashlib
import json

from pydantic import BaseModel, field_serializer

fps_log_interval = 10
status_report_interval = 10

class InputStatus(BaseModel):
    """Holds metrics for the input stream"""
    last_input_time: float | None = None
    fps: float = 0.0

    @field_serializer('last_input_time')
    def serialize_timestamps(self, v: float | None) -> int | None:
        return timestamp_to_ms(v)

class InferenceStatus(BaseModel):
    """Holds metrics for the inference process"""
    last_output_time: float | None = None
    fps: float = 0.0

    last_params_update_time: float | None = None
    last_params: dict | None = None
    last_params_hash: str | None = None

    last_error_time: float | None = None
    last_error: str | None = None

    last_restart_time: float | None = None
    last_restart_logs: list[str] | None = None
    restart_count: int = 0

    @field_serializer('last_output_time', 'last_params_update_time', 'last_error_time', 'last_restart_time')
    def serialize_timestamps(self, v: float | None) -> int | None:
        return timestamp_to_ms(v)

# Use a class instead of an enum since Pydantic can't handle serializing enums
class PipelineState:
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"
    DEGRADED_INPUT = "DEGRADED_INPUT"
    DEGRADED_INFERENCE = "DEGRADED_INFERENCE"
    ERROR = "ERROR"

class PipelineStatus(BaseModel):
    """Holds metrics for the pipeline streamer"""
    type: str = "status"
    pipeline: str
    start_time: float
    state: str = PipelineState.OFFLINE
    last_state_update_time: float | None = None

    input_status: InputStatus = InputStatus()
    inference_status: InferenceStatus = InferenceStatus()

    def update_params(self, params: dict, do_update_time=True):
        self.inference_status.last_params = params
        self.inference_status.last_params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if do_update_time:
            self.inference_status.last_params_update_time = time.time()
        return self

    @field_serializer('start_time', 'last_state_update_time')
    def serialize_timestamps(self, v: float | None) -> int | None:
        return timestamp_to_ms(v)

def timestamp_to_ms(v: float | None) -> int | None:
    return int(v * 1000) if v is not None else None

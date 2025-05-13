import asyncio
import logging
import time
from typing import Optional
import abc
from trickle import InputFrame, OutputFrame

from .process import PipelineProcess
from .status import PipelineState, PipelineStatus, InferenceStatus


class StreamerCallbacks(abc.ABC):
    """
    Interface implemented by the streamer for the ProcessGuardian to call back on.
    This is used to avoid circular dependencies between the ProcessGuardian and the Streamer.
    """

    @abc.abstractmethod
    def is_stream_running(self) -> bool: ...

    @abc.abstractmethod
    async def emit_monitoring_event(self, event_data: dict) -> None: ...

    @abc.abstractmethod
    def trigger_stop_stream(self) -> bool:
        """
        Trigger a stop of the stream. Returns True if the stream was actually
        stopped on this call or False if it was already stopped.
        """
        ...


class ProcessGuardian:
    """
    This class is responsible for keeping a pipeline process alive and monitoring its status.
    It also handles the streaming of input and output frames to the pipeline.
    """

    def __init__(
        self,
        pipeline: str,
        params: dict,
    ):
        self.pipeline = pipeline
        self.initial_params = params
        self.streamer: StreamerCallbacks = _NoopStreamerCallbacks()

        self.process: Optional[PipelineProcess] = None
        self.monitor_task = None
        self.status = PipelineStatus(pipeline=pipeline, start_time=0).update_params(
            params, False
        )

    async def start(self):
        self.process = PipelineProcess.start(self.pipeline, self.initial_params)
        self.status.update_state(PipelineState.INITIALIZING)
        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        if self.process:
            await self.process.stop()
            self.process = None

    async def reset_stream(
        self,
        request_id: str,
        stream_id: str,
        params: dict,
        streamer: StreamerCallbacks | None = None,
    ):
        if not self.process:
            raise RuntimeError("Process not running")
        self.status = PipelineStatus(pipeline=self.pipeline, start_time=time.time())
        self.streamer = streamer or _NoopStreamerCallbacks()
        self.process.reset_stream(request_id, stream_id)
        await self.update_params(params)

    def send_input(self, frame: InputFrame):
        if not self.process:
            raise RuntimeError("Process not running")
        iss = self.status.input_status
        if not iss.last_input_time:
            iss.last_input_time = time.time()
            # can't calculate fps from the first frame
        else:
            previous_input_time = max(iss.last_input_time, self.status.start_time)
            (iss.last_input_time, iss.fps) = calculate_rolling_fps(
                iss.fps, previous_input_time
            )

        self.process.send_input(frame)

    async def recv_output(self) -> OutputFrame | None:
        if not self.process:
            raise RuntimeError("Process not running")
        output = await self.process.recv_output()

        oss = self.status.inference_status
        if not oss.last_output_time:
            oss.last_output_time = time.time()
            # can't calculate fps from the first frame
        else:
            previous_output_time = max(oss.last_output_time, self.status.start_time)
            (oss.last_output_time, oss.fps) = calculate_rolling_fps(
                oss.fps, previous_output_time
            )

        return output

    async def update_params(self, params: dict):
        if not self.process:
            raise RuntimeError("Process not running")

        self.process.update_params(params)
        self.status.update_params(params)

        await self.streamer.emit_monitoring_event(
            {
                "type": "params_update",
                "pipeline": self.pipeline,
                "params": params,
                "params_hash": self.status.inference_status.last_params_hash,
                "update_time": self.status.inference_status.last_params_update_time,
            }
        )
        logging.info(
            f"ProcessGuardian: Parameter update queued. hash={self.status.inference_status.last_params_hash} params={params}"
        )

    def get_status(self, clear_transient: bool = False) -> PipelineStatus:
        status = self.status.model_copy(deep=True)
        if clear_transient:
            # Clear the large transient fields if requested, but do return them
            self.status.inference_status.last_params = None
            self.status.inference_status.last_restart_logs = None
        return status

    def _compute_current_state(self) -> str:
        # Special case: process not running or initializing
        if not self.process or not self.process.is_alive():
            logging.error("Process is not alive. Returning ERROR state")
            return PipelineState.ERROR
        elif not self.process.is_pipeline_initialized() or self.process.done.is_set():
            # done is only set in the middle of the restart process so also return INITIALIZING
            return PipelineState.INITIALIZING

        # Special case: stream shutdown
        current_time = time.time()
        input = self.status.input_status
        time_since_last_input = current_time - (input.last_input_time or 0)

        if not self.streamer.is_stream_running():
            return (
                PipelineState.OFFLINE
                if time_since_last_input > 3  # 3s grace period after shutdown
                else PipelineState.DEGRADED_INPUT
            )
        elif time_since_last_input > 60:
            if self.streamer.trigger_stop_stream():
                logging.info(
                    f"Shutting down streamer. Flagging DEGRADED_INPUT state during shutdown: time_since_last_input={time_since_last_input:.1f}s"
                )
            return (
                PipelineState.DEGRADED_INPUT
                if time_since_last_input < 90
                else PipelineState.ERROR  # Not shutting down after 30s, declare ERROR
            )

        # Special case: pipeline load
        start_time = max(self.process.start_time, self.status.start_time)
        inference = self.status.inference_status

        time_since_last_output = current_time - (inference.last_output_time or 0)
        pipeline_load_time = max(inference.last_params_update_time or 0, start_time)
        # -1s to be conservative and avoid race conditions on the timestamp comparisons below
        time_since_pipeline_load = max(0, current_time - pipeline_load_time - 1)

        active_after_load = time_since_last_output < time_since_pipeline_load
        if not active_after_load:
            is_params_update = (inference.last_params_update_time or 0) > start_time
            load_grace_period = 2 if is_params_update else 10
            load_timeout = 30 if is_params_update else 120
            return (
                PipelineState.ONLINE
                if time_since_pipeline_load < load_grace_period
                else PipelineState.DEGRADED_INPUT
                if time_since_last_input > time_since_pipeline_load
                else PipelineState.DEGRADED_INFERENCE
                if time_since_pipeline_load < load_timeout
                else PipelineState.ERROR  # Not starting after timeout, declare ERROR
            )

        # Normal case: active stream
        stopped_producing_frames = (
            time_since_last_output > (time_since_last_input + 1)
            and time_since_last_output > 5
        )
        if stopped_producing_frames:
            return PipelineState.ERROR

        recent_error = (inference.last_error_time or 0) > current_time - 15
        recent_restart = (inference.last_restart_time or 0) > current_time - 60
        if recent_error or recent_restart:
            return PipelineState.DEGRADED_INFERENCE
        elif time_since_last_input > 2 or input.fps < 15:
            return PipelineState.DEGRADED_INPUT
        elif time_since_last_output > 2 or inference.fps < min(10, 0.8 * input.fps):
            return PipelineState.DEGRADED_INFERENCE

        return PipelineState.ONLINE

    async def _restart_process(self):
        if not self.process:
            raise RuntimeError("Process not started")

        # Restarting the process will take a couple of time, so we stop the stream
        # before it happens so the gateway/app can switch to a functioning O ASAP.
        logging.info(
            f"Stopping streamer due to process restart prev_restart_count={self.status.inference_status.restart_count}"
        )
        self.streamer.trigger_stop_stream()

        # Capture logs before stopping the process
        restart_logs = self.process.get_recent_logs()
        # don't call the full start/stop methods since we only want to restart the process
        await self.process.stop()

        self.process = PipelineProcess.start(self.pipeline, self.initial_params)
        self.status.update_state(PipelineState.INITIALIZING)
        curr_status = self.status.inference_status
        self.status.inference_status = InferenceStatus(
            restart_count=curr_status.restart_count + 1,
            last_restart_time=time.time(),
            last_restart_logs=restart_logs,
        )

        await self.streamer.emit_monitoring_event(
            {
                "type": "restart",
                "pipeline": self.pipeline,
                "restart_count": self.status.inference_status.restart_count,
                "restart_time": self.status.inference_status.last_restart_time,
                "restart_logs": restart_logs,
            }
        )

        logging.info(
            f"PipelineProcess restarted. Restart count: {self.status.inference_status.restart_count}"
        )

    async def _monitor_loop(self):
        while True:
            try:
                await asyncio.sleep(1)
                if not self.process:
                    continue

                last_error = self.process.get_last_error()
                if last_error:
                    error_msg, error_time = last_error
                    self.status.inference_status.last_error = error_msg
                    self.status.inference_status.last_error_time = error_time
                    await self.streamer.emit_monitoring_event(
                        {
                            "type": "error",
                            "pipeline": self.pipeline,
                            "message": error_msg,
                            "time": error_time,
                        }
                    )

                state = self._compute_current_state()
                if state != self.status.state:
                    time_since_last_state_update = time.time() - (
                        self.status.last_state_update_time or 0
                    )
                    logging.info(
                        f"Pipeline state changed. old_state={self.status.state} new_state={state} time_since_last_state_update={time_since_last_state_update:.1f}s "
                        + f"process_alive={self.process.is_alive()} process_initialized={self.process.is_pipeline_initialized()} process_done={self.process.done.is_set()} "
                        + f"streamer_running={self.streamer.is_stream_running()} status={self.status.model_dump_json()}"
                    )
                    self.status.update_state(state)

                if state == PipelineState.ERROR:
                    if self.status.inference_status.restart_count >= 3:
                        logging.error(
                            "Pipeline process max restarts reached. Staying in ERROR state"
                        )
                        continue
                    logging.error("Pipeline is in ERROR state. Restarting process.")
                    # Hot fix: the comfyui pipeline process is having trouble shutting down and causes restarts not to recover.
                    # So we skip the restart here and leave the status in ERROR so the worker will restart the whole container.
                    logging.warning("Skipping process restart, staying in ERROR state")
                    # TODO: Uncomment this once pipeline shutdown is fixed and restarting process is useful again.
                    # await self._restart_process()
            except asyncio.CancelledError:
                return
            except Exception:
                logging.exception("Error in monitor loop", stack_info=True)
                continue


fps_ema_alpha = 0.0645  # 2 + (30 + 1); to give the most weight to the past 30 frames


def calculate_rolling_fps(previous_fps: float, previous_frame_time: float):
    now = time.time()
    time_since_last_frame = now - previous_frame_time
    if time_since_last_frame <= 0:
        return (now, previous_fps)  # Avoid division by zero or negative time
    current_fps = 1 / time_since_last_frame
    new_fps = fps_ema_alpha * current_fps + (1 - fps_ema_alpha) * previous_fps
    return (now, new_fps)


class _NoopStreamerCallbacks(StreamerCallbacks):
    async def is_stream_running(self) -> bool:
        return False

    async def emit_monitoring_event(self, event_data: dict) -> None:
        pass

    def trigger_stop_stream(self) -> bool:
        return False

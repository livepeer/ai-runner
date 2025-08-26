import asyncio
import logging
import time
from typing import Optional
import abc

from trickle import InputFrame, OutputFrame
from .process import PipelineProcess
from .status import PipelineState, PipelineStatus, InferenceStatus, InputStatus

FPS_LOG_INTERVAL = 10.0

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

        self.input_fps_counter = FPSCounter()
        self.output_fps_counter = FPSCounter()

    async def start(self):
        self.process = PipelineProcess.start(self.pipeline, self.initial_params)
        self.status.update_state(PipelineState.LOADING)
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
        manifest_id: str,
        stream_id: str,
        params: dict,
        streamer: StreamerCallbacks | None = None,
    ):
        if not self.process:
            raise RuntimeError("Process not running")
        self.status.start_time = time.time()
        self.status.input_status = InputStatus()
        self.input_fps_counter.reset()
        self.output_fps_counter.reset()
        self.streamer = streamer or _NoopStreamerCallbacks()

        self.process.reset_stream(request_id, manifest_id, stream_id)
        await self.update_params(params)
        self.status.update_state(PipelineState.ONLINE)

    def send_input(self, frame: InputFrame):
        if not self.process:
            raise RuntimeError("Process not running")

        self.status.input_status.last_input_time = time.time()
        self.input_fps_counter.inc()

        self.process.send_input(frame)

    async def recv_output(self) -> OutputFrame | None:
        if not self.process:
            raise RuntimeError("Process not running")
        output = await self.process.recv_output()

        self.status.inference_status.last_output_time = time.time()
        self.output_fps_counter.inc()

        return output

    async def update_params(self, params: dict):
        if not self.process:
            raise RuntimeError("Process not running")

        self.process.update_params(params)
        self.status.update_params(params)

        # The gateway is having trouble keeping up with a high frequency of events, while we still want to support high
        # frequency of parameter updates. So we're skipping the event for now.
        # TODO: Re-enable this once the trickle protocol is udpated to support higher event throughput (e.g. #720)
        # await self.streamer.emit_monitoring_event(
        #     {
        #         "type": "params_update",
        #         "pipeline": self.pipeline,
        #         "params": params,
        #         "params_hash": self.status.inference_status.last_params_hash,
        #         "update_time": self.status.inference_status.last_params_update_time,
        #     }
        # )
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
        if self.status.state == PipelineState.ERROR:
            # The ERROR state should be permanent, so ignore the other checks below
            return PipelineState.ERROR

        # Special case: process not running or initializing
        if not self.process or not self.process.is_alive():
            logging.error("Process is not alive. Returning ERROR state")
            return PipelineState.ERROR
        elif not self.process.is_pipeline_initialized() or self.process.done.is_set():
            # done is only set in the middle of the restart process so also return INITIALIZING
            return PipelineState.LOADING

        # Special case: stream not running
        current_time = time.time()
        input = self.status.input_status
        start_time = max(self.process.start_time, self.status.start_time)
        time_since_last_input = current_time - (input.last_input_time or start_time)

        if not self.streamer.is_stream_running():
            # give it 3s `DEGRADED_INPUT` grace period after shutdown
            is_offline = time_since_last_input > 3 or not input.last_input_time
            return PipelineState.OFFLINE if is_offline else PipelineState.DEGRADED_INPUT

        # Special case: stream shutdown after inactivity
        if time_since_last_input > 60:
            if self.streamer.trigger_stop_stream():
                logging.info(
                    f"Shutting down streamer. Flagging DEGRADED_INPUT state during shutdown: time_since_last_input={time_since_last_input:.1f}s"
                )
            if time_since_last_input < 90:
                return PipelineState.DEGRADED_INPUT
            # Stream hasn't shut down 30s after the trigger above (90s total idle time).
            # Declare ERROR so the container gets restarted by the worker.
            return PipelineState.ERROR

        # Special case: pipeline load
        inference = self.status.inference_status
        time_since_last_output = current_time - (inference.last_output_time or 0)
        pipeline_load_time = max(inference.last_params_update_time or 0, start_time)
        # -2s to be conservative and avoid race conditions on the timestamp comparisons below
        time_since_pipeline_load = max(0, current_time - pipeline_load_time - 2)

        active_after_load = time_since_last_output < time_since_pipeline_load
        logging.debug(f"[_compute_current_state] active_after_load={active_after_load} time_since_last_output={time_since_last_output:.1f}s time_since_pipeline_load={time_since_pipeline_load:.1f}s")
        if not active_after_load:
            is_params_update = (inference.last_params_update_time or 0) > start_time
            load_grace_period = 2 if is_params_update else 10
            load_timeout = 60 if is_params_update else 120
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

        # Capture logs before stopping the process
        restart_logs = self.process.get_recent_logs()
        # don't call the full start/stop methods since we only want to restart the process
        await self.process.stop()

        self.process = PipelineProcess.start(self.pipeline, self.initial_params)
        self.status.update_state(PipelineState.LOADING)
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
        last_fps_compute = time.time()
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

                now = time.time()
                if now - last_fps_compute > FPS_LOG_INTERVAL:
                    input_fps = self.input_fps_counter.fps(now)
                    output_fps = self.output_fps_counter.fps(now)

                    self.status.input_status.fps, self.status.inference_status.fps = input_fps, output_fps
                    last_fps_compute = now

                    if self.streamer.is_stream_running():
                        if input_fps > 0:
                            logging.info(f"Input FPS: {input_fps:.2f}")
                        if output_fps > 0:
                            logging.info(f"Output FPS: {output_fps:.2f}")

                state = self._compute_current_state()
                if state == self.status.state:
                    continue

                if state == PipelineState.OFFLINE:
                    # Revert to initial params when the stream stops
                    self.process.update_params(self.initial_params)

                if state != PipelineState.ERROR:
                    # avoid thrashing the state to ERROR if we're going to restart the process below
                    self.status.update_state(state)
                else:
                    try:
                        restart_count = self.status.inference_status.restart_count
                        logging.error(
                            f"Pipeline is in ERROR state. Stopping streamer and restarting process. prev_restart_count={restart_count}"
                        )

                        # Restarting the process to fix the error will take a couple of time, so we also stop
                        # the stream before it happens so the gateway/app can switch to a functioning O ASAP.
                        self.streamer.trigger_stop_stream()

                        if restart_count >= 3:
                            raise Exception(f"Pipeline process max restarts reached ({restart_count})")

                        if self.pipeline == "comfyui":
                            # Hot fix: the comfyui pipeline process is having trouble shutting down and causes restarts not to recover.
                            # So we skip the restart here and move the state to ERROR so the worker will restart the whole container.
                            # TODO: Remove this exception once pipeline shutdown is fixed and restarting process is useful again.
                            raise Exception(
                                "Skipping process restart due to pipeline shutdown issues"
                            )
                        await self._restart_process()
                    except Exception:
                        logging.exception("Failed to stop streamer and restart process. Moving to ERROR state", stack_info=True)
                        self.status.update_state(PipelineState.ERROR)


            except asyncio.CancelledError:
                return
            except Exception:
                logging.exception("Error in monitor loop", stack_info=True)
                continue

class FPSCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the counter to initial state. Should only be called in between streams.
        """
        self.frame_count = 0
        self.start_time = None

    def inc(self):
        if not self.start_time:
            # first frame of the stream only sets the measurement window start time
            self.start_time = time.time()
            self.frame_count = 0
        else:
            self.frame_count += 1

    def fps(self, now = time.time()) -> float:
        if not self.frame_count or not self.start_time:
            fps = 0
        else:
            fps = self.frame_count / (now - self.start_time)

        # start next measuring window immediately, do not wait for the next frame
        self.start_time = now
        self.frame_count = 0

        return fps

class _NoopStreamerCallbacks(StreamerCallbacks):
    def is_stream_running(self) -> bool:
        return False

    async def emit_monitoring_event(self, event_data: dict) -> None:
        pass

    def trigger_stop_stream(self) -> bool:
        return False

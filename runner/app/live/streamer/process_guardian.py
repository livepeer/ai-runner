import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional
from trickle import InputFrame, OutputFrame

from .process import PipelineProcess
from .status import PipelineState, PipelineStatus


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
        self.params = params
        self.monitoring_callback = _noop_callback

        self.process = None
        self.monitor_task = None
        self.status = PipelineStatus(
            pipeline=pipeline, start_time=time.time()
        ).update_params(params, False)

    async def start(self):
        # Start the pipeline process and initialize timing
        self.process = PipelineProcess.start(self.pipeline, self.params)
        if self.process is None:
            raise RuntimeError("Failed to start PipelineProcess")
        # Launch the monitor loop as a background task
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
        monitoring_callback: Callable[[dict], Awaitable[None]],
    ):
        if not self.process:
            raise RuntimeError("Process not running")
        self.status = PipelineStatus(pipeline=self.pipeline, start_time=time.time())
        self.monitoring_callback = monitoring_callback
        self.process.reset_stream(request_id, stream_id)
        await self.update_params(params)

    def send_input(self, frame: InputFrame):
        if not self.process:
            raise RuntimeError("Process not running")
        iss = self.status.input_status
        previous_input_time = max(iss.last_input_time or 0, self.process.start_time)
        (iss.last_input_time, iss.fps) = calculate_rolling_fps(
            iss.fps, previous_input_time
        )

        self.process.send_input(frame)

    async def recv_output(self) -> OutputFrame | None:
        if not self.process:
            raise RuntimeError("Process not running")
        output = await self.process.recv_output()

        oss = self.status.inference_status
        previous_output_time = max(oss.last_output_time or 0, self.process.start_time)
        (oss.last_output_time, oss.fps) = calculate_rolling_fps(
            oss.fps, previous_output_time
        )
        return output

    async def update_params(self, params: dict):
        if not self.process:
            raise RuntimeError("Process not running")
        self.params = params
        logging.info(f"ProcessGuardian: Queuing parameter update with hash={self.status.inference_status.last_params_hash}, params={params}")
        self.process.update_params(params)
        self.status.update_params(params)

        await self.monitoring_callback(
            {
                "type": "params_update",
                "pipeline": self.pipeline,
                "params": params,
                "params_hash": self.status.inference_status.last_params_hash,
                "update_time": self.status.inference_status.last_params_update_time,
            }
        )
        logging.info(f"ProcessGuardian: Parameter update queued and monitoring callback completed. Hash={self.status.inference_status.last_params_hash}")

    def get_status(self, clear_transient: bool = False) -> PipelineStatus:
        new_state = self._current_state()
        if new_state != self.status.state:
            self.status.state = new_state
            self.status.last_state_update_time = time.time()
            logging.info(f"Pipeline state changed to {new_state}")
        status = self.status.model_copy()
        if clear_transient:
            # Clear the large transient fields if requested, but do return them
            self.status.inference_status.last_params = None
            self.status.inference_status.last_restart_logs = None
        return status

    def _current_state(self) -> str:
        current_time = time.time()
        input = self.status.input_status
        last_input_time = input.last_input_time or 0
        time_since_last_input = current_time - last_input_time
        if time_since_last_input > 60:
            if time_since_last_input < 90:
                # streamer should stop automatically after 60s, so give ourselves a 30s grace period to shutdown
                logging.info(
                    f"Detected DEGRADED_INPUT during shutdown: time_since_last_input={time_since_last_input:.1f}s"
                )
                return PipelineState.DEGRADED_INPUT
            return PipelineState.OFFLINE
        elif time_since_last_input > 2 or input.fps < 15:
            logging.info(
                f"Detected DEGRADED_INPUT: time_since_last_input={time_since_last_input:.1f}s, fps={input.fps}"
            )
            return PipelineState.DEGRADED_INPUT

        inference = self.status.inference_status
        pipeline_load_time = max(
            self.status.start_time, inference.last_params_update_time or 0
        )
        if inference.last_output_time and current_time - pipeline_load_time < 30:
            # 30s grace period for the pipeline to start
            return PipelineState.ONLINE

        delayed_frames = (
            not inference.last_output_time
            or current_time - inference.last_output_time > 5
        )
        low_fps = inference.fps < min(10, 0.8 * input.fps)
        recent_restart = (
            inference.last_restart_time
            and current_time - inference.last_restart_time < 60
        )
        recent_error = (
            inference.last_error_time and current_time - inference.last_error_time < 15
        )
        if delayed_frames or low_fps or recent_restart or recent_error:
            return PipelineState.DEGRADED_INFERENCE

        return PipelineState.ONLINE

    async def _restart_process(self):
        if not self.process:
            raise RuntimeError("Process not started")

        # Capture logs before stopping the process
        restart_logs = self.process.get_recent_logs()
        last_error = self.process.get_last_error()
        # don't call the full start/stop methods since we only want to restart the process
        await self.process.stop()

        self.process = PipelineProcess.start(self.pipeline, self.params)
        self.status.inference_status.restart_count += 1
        self.status.inference_status.last_restart_time = time.time()
        self.status.inference_status.last_restart_logs = restart_logs
        if last_error:
            error_msg, error_time = last_error
            self.status.inference_status.last_error = error_msg
            self.status.inference_status.last_error_time = error_time

        await self.monitoring_callback(
            {
                "type": "restart",
                "pipeline": self.pipeline,
                "restart_count": self.status.inference_status.restart_count,
                "restart_time": self.status.inference_status.last_restart_time,
                "restart_logs": restart_logs,
                "last_error": last_error,
            }
        )

        logging.info(
            f"PipelineProcess restarted. Restart count: {self.status.inference_status.restart_count}"
        )

    async def _monitor_loop(self):
        try:
            while True:
                await asyncio.sleep(1)
                if not self.process or self.process.done.is_set():
                    continue

                last_error = self.process.get_last_error()
                if last_error:
                    error_msg, error_time = last_error
                    self.status.inference_status.last_error = error_msg
                    self.status.inference_status.last_error_time = error_time
                    await self.monitoring_callback(
                        {
                            "type": "error",
                            "pipeline": self.pipeline,
                            "error": error_msg,
                            "time": error_time,
                        }
                    )

                start_time = self.process.start_time
                current_time = time.time()
                last_input_time = max(
                    self.status.input_status.last_input_time or 0, start_time
                )
                last_output_time = max(
                    self.status.inference_status.last_output_time or 0, start_time
                )
                last_params_update_time = max(
                    self.status.inference_status.last_params_update_time or 0,
                    start_time,
                )

                time_since_last_input = current_time - last_input_time
                time_since_last_output = current_time - last_output_time
                time_since_start = current_time - start_time
                time_since_last_params = current_time - last_params_update_time
                time_since_reload = min(time_since_last_params, time_since_start)

                gone_stale = (
                    time_since_last_output > time_since_last_input
                    and time_since_last_output > 60
                    and time_since_reload > 240
                )
                if time_since_last_input > 5 and not gone_stale:
                    # nothing to do if we're not sending inputs
                    continue

                active_after_reload = time_since_last_output < (time_since_reload - 1)
                stopped_recently = (
                    active_after_reload
                    and time_since_last_output > 5
                    and time_since_last_output < 60
                )
                if stopped_recently or gone_stale:
                    logging.warning(
                        "No output received while inputs are being sent. Restarting process."
                    )
                    await self._restart_process()
        except asyncio.CancelledError:
            pass


fps_ema_alpha = 0.0645  # 2 + (30 + 1); to give the most weight to the past 30 frames


def calculate_rolling_fps(previous_fps: float, previous_frame_time: float):
    now = time.time()
    time_since_last_frame = now - previous_frame_time
    if time_since_last_frame <= 0:
        return (now, previous_fps)  # Avoid division by zero or negative time
    current_fps = 1 / time_since_last_frame
    new_fps = fps_ema_alpha * current_fps + (1 - fps_ema_alpha) * previous_fps
    return (now, new_fps)


async def _noop_callback(_):
    pass

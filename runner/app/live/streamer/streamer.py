import asyncio
import logging
import os
import time
import numpy as np
from multiprocessing.synchronize import Event
from typing import AsyncGenerator, Awaitable
from asyncio import Lock

import cv2
from PIL import Image

from .process import PipelineProcess
from .protocol.protocol import StreamProtocol
from .status import PipelineStatus, PipelineState, timestamp_to_ms
from trickle import AudioFrame, VideoFrame, OutputFrame, AudioOutput, VideoOutput

fps_log_interval = 10
status_report_interval = 10

class PipelineStreamer:
    def __init__(self, protocol: StreamProtocol, pipeline: str, input_timeout: int, params: dict):
        self.protocol = protocol
        self.pipeline = pipeline
        self.params = params
        self.input_timeout = input_timeout  # 0 means disabled

        self.status = PipelineStatus(pipeline=pipeline, start_time=time.time()).update_params(params, False)
        self.stop_event = asyncio.Event()
        self.emit_event_lock = Lock()
        self.process: PipelineProcess | None = None

        self.main_tasks: list[asyncio.Task] = []
        self.tasks_supervisor_task: asyncio.Task | None = None

    async def start(self):
        if self.process:
            raise RuntimeError("Streamer already started")

        self.stop_event.clear()
        self.process = PipelineProcess.start(self.pipeline, self.params)
        await self.protocol.start()

        # We need a bunch of concurrent tasks to run the streamer. So we start them all in background and then also start
        # a supervisor task that will stop everything if any of the main tasks return or the stop event is set.
        self.main_tasks = [
            run_in_background("ingress_loop", self.run_ingress_loop()),
            run_in_background("egress_loop", self.run_egress_loop()),
            run_in_background("monitor_loop", self.monitor_loop()),
            run_in_background("control_loop", self.run_control_loop()),
            run_in_background("report_status_loop", self.report_status_loop())
        ]
        self.tasks_supervisor_task = run_in_background("tasks_supervisor", self.tasks_supervisor())

    async def tasks_supervisor(self):
        """Supervises the main tasks and stops everything if either of them return or the stop event is set"""
        try:
            async def wait_for_stop():
                await self.stop_event.wait()

            tasks = self.main_tasks + [asyncio.create_task(wait_for_stop())]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            await self._do_stop()
        except Exception:
            logging.error("Error on supervisor task", exc_info=True)
            os._exit(1)

    async def _do_stop(self):
        """Stops all running tasks and waits for them to exit. To be called only by the supervisor task"""
        if not self.process:
            raise RuntimeError("Process not started")

        # make sure the stop event is set and give running tasks a chance to exit cleanly
        self.stop_event.set()
        _, pending = await asyncio.wait(self.main_tasks, return_when=asyncio.ALL_COMPLETED, timeout=1)
        # force cancellation of the remaining tasks
        for task in pending:
            task.cancel()

        await asyncio.gather(self.protocol.stop(), self.process.stop(), return_exceptions=True)

    async def wait(self):
        if not self.process:
            raise RuntimeError("Streamer not started")
        return await self.tasks_supervisor_task

    async def stop(self, *, timeout: float):
        self.stop_event.set()
        await asyncio.wait_for(self.tasks_supervisor_task, timeout)

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

        await self._emit_monitoring_event({
            "type": "restart",
            "pipeline": self.pipeline,
            "restart_count": self.status.inference_status.restart_count,
            "restart_time": self.status.inference_status.last_restart_time,
            "restart_logs": restart_logs,
            "last_error": last_error
        })

        logging.info(
            f"PipelineProcess restarted. Restart count: {self.status.inference_status.restart_count}"
        )

    async def update_params(self, params: dict):
        self.params = params
        if self.process:
            self.process.update_params(params)
        self.status.update_params(params)

        await self._emit_monitoring_event({
            "type": "params_update",
            "pipeline": self.pipeline,
            "params": params,
            "params_hash": self.status.inference_status.last_params_hash,
            "update_time": self.status.inference_status.last_params_update_time
        })

    async def report_status_loop(self):
        next_report = time.time() + status_report_interval
        while not self.stop_event.is_set():
            current_time = time.time()
            if next_report <= current_time:
                # If we lost track of the next report time, just report immediately
                next_report = current_time + status_report_interval
            else:
                await asyncio.sleep(next_report - current_time)
                next_report += status_report_interval

            event = self.get_status().model_dump()
            await self._emit_monitoring_event(event)

            # Clear the large transient fields after reporting them once
            self.status.inference_status.last_params = None
            self.status.inference_status.last_restart_logs = None

    def get_status(self) -> PipelineStatus:
        new_state = self._current_state()
        if new_state != self.status.state:
            self.status.state = new_state
            self.status.last_state_update_time = time.time()
            logging.info(f"Pipeline state changed to {new_state}")
        return self.status.model_copy()

    def _current_state(self) -> str:
        current_time = time.time()
        input = self.status.input_status
        last_input_time = input.last_input_time or 0
        if current_time - last_input_time > 60:
            if self.stop_event.is_set() and current_time - last_input_time < 90:
                # give ourselves a 30s grace period to shutdown
                return PipelineState.DEGRADED_INPUT
            return PipelineState.OFFLINE
        elif current_time - last_input_time > 2 or input.fps < 15:
            return PipelineState.DEGRADED_INPUT

        inference = self.status.inference_status
        pipeline_load_time = max(self.status.start_time, inference.last_params_update_time or 0)
        if not inference.last_output_time and current_time - pipeline_load_time < 30:
            # 30s grace period for the pipeline to start
            return PipelineState.ONLINE

        delayed_frames = not inference.last_output_time or current_time - inference.last_output_time > 5
        low_fps = inference.fps < min(10, 0.8 * input.fps)
        recent_restart = inference.last_restart_time and current_time - inference.last_restart_time < 60
        recent_error = inference.last_error_time and current_time - inference.last_error_time < 15
        if delayed_frames or low_fps or recent_restart or recent_error:
            return PipelineState.DEGRADED_INFERENCE

        return PipelineState.ONLINE

    async def _emit_monitoring_event(self, event: dict):
        """Protected method to emit monitoring event with lock"""
        event["timestamp"] = timestamp_to_ms(time.time())
        logging.info(f"Emitting monitoring event: {event}")
        async with self.emit_event_lock:
            try:
                await self.protocol.emit_monitoring_event(event)
            except Exception as e:
                logging.error(f"Failed to emit monitoring event: {e}")

    async def monitor_loop(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(1)
            if not self.process or self.process.done.is_set():
                continue

            last_error = self.process.get_last_error()
            if last_error:
                error_msg, error_time = last_error
                self.status.inference_status.last_error = error_msg
                self.status.inference_status.last_error_time = error_time
                await self._emit_monitoring_event({
                    "type": "error",
                    "pipeline": self.pipeline,
                    "error": error_msg,
                    "time": error_time
                })

            start_time = self.process.start_time
            current_time = time.time()
            last_input_time = max(self.status.input_status.last_input_time or 0, start_time)
            last_output_time = max(self.status.inference_status.last_output_time or 0, start_time)
            last_params_update_time = max(self.status.inference_status.last_params_update_time or 0, start_time)

            time_since_last_input = current_time - last_input_time
            time_since_last_output = current_time - last_output_time
            time_since_start = current_time - start_time
            time_since_last_params = current_time - last_params_update_time
            time_since_reload = min(time_since_last_params, time_since_start)

            if self.input_timeout > 0 and time_since_last_input > self.input_timeout:
                logging.info(f"Input stream stopped for {time_since_last_input} seconds. Shutting down...")
                self.stop_event.set()
                return

            gone_stale = (
                time_since_last_output > time_since_last_input
                and time_since_last_output > 60
                and time_since_reload > 240
            )
            if time_since_last_input > 5 and not gone_stale:
                # nothing to do if we're not sending inputs
                continue

            active_after_reload = time_since_last_output < (time_since_reload - 1)
            stopped_recently = active_after_reload and time_since_last_output > 5 and time_since_last_output < 60
            if stopped_recently or gone_stale:
                logging.warning(
                    "No output received while inputs are being sent. Restarting process."
                )
                await self._restart_process()

    async def run_ingress_loop(self):
        frame_count = 0
        start_time = 0.0
        async for av_frame in self.protocol.ingress_loop(self.stop_event):
            if not self.process or self.process.done.is_set():
                # no need to sleep since we want to consume input frames as fast as possible
                continue

            if not start_time:
                start_time = time.time()

            # TODO any necessary accounting here for audio
            if isinstance(av_frame, AudioFrame):
                self.process.send_input(av_frame)
                continue

            if not isinstance(av_frame, VideoFrame):
                logging.warning("Unknown frame type received, dropping")
                continue

            frame = av_frame.image
            if not frame:
                continue

            if frame.mode != "RGBA":
                frame = frame.convert("RGBA")

            # crop the max square from the center of the image and scale to 512x512
            # most models expect this size especially when using tensorrt
            width, height = frame.size
            if (width, height) != (512, 512):
                frame_array = np.array(frame)

                if width != height:
                    square_size = min(width, height)
                    start_x = width // 2 - square_size // 2
                    start_y = height // 2 - square_size // 2
                    frame_array = frame_array[start_y:start_y+square_size, start_x:start_x+square_size]

                # Resize using cv2 (much faster than PIL)
                frame_array = cv2.resize(frame_array, (512, 512))
                frame = Image.fromarray(frame_array)

            logging.debug(f"Sending input frame. Scaled from {width}x{height} to {frame.size[0]}x{frame.size[1]}")
            av_frame = av_frame.replace_image(frame)
            self.process.send_input(av_frame)
            self.status.input_status.last_input_time = time.time()  # Track time after send completes

            # Increment frame count and measure FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_log_interval:
                self.status.input_status.fps = frame_count / elapsed_time
                logging.info(f"Input FPS: {self.status.input_status.fps:.2f}")
                frame_count = 0
                start_time = time.time()
        logging.info("Ingress loop ended")

    async def run_egress_loop(self):
        async def gen_output_frames() -> AsyncGenerator[OutputFrame, None]:
            frame_count = 0
            start_time = 0.0
            while not self.stop_event.is_set():
                if not self.process or self.process.done.is_set():
                    await asyncio.sleep(0.05)
                    continue

                output_frame = await self.process.recv_output()
                if not output_frame:
                    continue

                # TODO accounting for audio output
                if isinstance(output_frame, AudioOutput):
                    yield output_frame
                    continue

                if not isinstance(output_frame, VideoOutput):
                    logging.warning(f"Unknown output frame type {type(output_frame)}, dropping")
                    continue

                output_image = output_frame.frame.image

                if not start_time:
                    # only start measuring output FPS after the first frame
                    start_time = time.time()

                self.status.inference_status.last_output_time = time.time()  # Track time after receive completes
                logging.debug(
                    f"Output image received out_width: {output_image.width}, out_height: {output_image.height}"
                )

                yield output_frame

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    self.status.inference_status.fps = frame_count / elapsed_time
                    logging.info(f"Output FPS: {self.status.inference_status.fps:.2f}")
                    frame_count = 0
                    start_time = time.time()

        await self.protocol.egress_loop(gen_output_frames())
        logging.info("Egress loop ended")

    async def run_control_loop(self):
        """Consumes control messages from the protocol and updates parameters"""
        async for params in self.protocol.control_loop(self.stop_event):
            try:
                await self.update_params(params)
            except Exception as e:
                logging.error(f"Error updating model with control message: {e}")
        logging.info("Control loop ended")


def run_in_background(task_name: str, coro: Awaitable):
    async def task_wrapper():
        try:
            await coro
        except Exception as e:
            logging.error(f"Error in task {task_name}", exc_info=True)

    return asyncio.create_task(task_wrapper())

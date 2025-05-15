import asyncio
import logging
import os
import time
import numpy as np
from typing import AsyncGenerator, Awaitable
from asyncio import Lock

import cv2
from PIL import Image

from .process_guardian import ProcessGuardian, ProcessCallbacks
from .protocol.protocol import StreamProtocol
from .status import timestamp_to_ms
from trickle import AudioFrame, VideoFrame, OutputFrame, AudioOutput, VideoOutput

fps_log_interval = 10
status_report_interval = 10

class PipelineStreamer(ProcessCallbacks):
    def __init__(
        self,
        protocol: StreamProtocol,
        input_timeout: int,
        process: ProcessGuardian,
        request_id: str,
        manifest_id: str,
        stream_id: str,
    ):
        self.protocol = protocol
        self.input_timeout = input_timeout  # 0 means disabled
        self.process = process

        self.stop_event = asyncio.Event()
        self.emit_event_lock = Lock()

        self.main_tasks: list[asyncio.Task] = []
        self.tasks_supervisor_task: asyncio.Task | None = None
        self.request_id = request_id
        self.manifest_id = manifest_id
        self.stream_id = stream_id

    async def start(self, params: dict):
        if self.tasks_supervisor_task:
            raise RuntimeError("Streamer already started")

        await self.process.reset_stream(
            self.request_id, self.manifest_id, self.stream_id, params, self
        )

        self.stop_event.clear()
        await self.protocol.start()

        # We need a bunch of concurrent tasks to run the streamer. So we start them all in background and then also start
        # a supervisor task that will stop everything if any of the main tasks return or the stop event is set.
        self.main_tasks = [
            run_in_background("ingress_loop", self.run_ingress_loop()),
            run_in_background("egress_loop", self.run_egress_loop()),
            run_in_background("report_status_loop", self.report_status_loop()),
            run_in_background("control_loop", self.run_control_loop()),
        ]
        # auxiliary tasks that are not critical to the supervisor, but which we want to run
        # TODO: maybe remove this since we had to move the control loop to main tasks
        self.auxiliary_tasks: list[asyncio.Task] = []
        self.tasks_supervisor_task = run_in_background(
            "tasks_supervisor", self.tasks_supervisor()
        )

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
        if not self.tasks_supervisor_task:
            raise RuntimeError("Process not started")

        # make sure the stop event is set and give running tasks a chance to exit cleanly
        self.stop_event.set()
        _, pending = await asyncio.wait(
            self.main_tasks + self.auxiliary_tasks,
            return_when=asyncio.ALL_COMPLETED,
            timeout=1,
        )
        # force cancellation of the remaining tasks
        for task in pending:
            task.cancel()

        await asyncio.gather(self.protocol.stop(), return_exceptions=True)
        self.main_tasks = []
        self.auxiliary_tasks = []
        self.tasks_supervisor_task = None
        self.process.stop_stream()

    async def wait(self):
        if not self.tasks_supervisor_task:
            raise RuntimeError("Streamer not started")
        return await self.tasks_supervisor_task

    def is_running(self):
        return self.tasks_supervisor_task is not None

    async def stop(self, *, timeout: float):
        if not self.tasks_supervisor_task:
            raise RuntimeError("Streamer not started")
        self.stop_event.set()
        await asyncio.wait_for(asyncio.shield(self.tasks_supervisor_task), timeout)

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

            status = self.process.get_status(clear_transient=True)
            await self.emit_monitoring_event(status.model_dump())

            last_input_time = max(
                status.input_status.last_input_time or 0, status.start_time
            )
            time_since_last_input = time.time() - last_input_time
            if self.input_timeout > 0 and time_since_last_input > self.input_timeout:
                logging.info(
                    f"Input stream stopped for {time_since_last_input} seconds. Shutting down..."
                )
                self.stop_event.set()

    def on_before_process_restart(self, restart_count: int) -> None:
        # Restarting the process will take a couple of time, so we stop the stream
        # before it happens so the gateway/app can switch to a functioning O ASAP.
        logging.info(f"Stopping streamer due to process restart restart_count={restart_count}")
        self.stop_event.set()

    async def emit_monitoring_event(self, event: dict, queue_event_type: str = "ai_stream_events"):
        """Protected method to emit monitoring event with lock"""
        event["timestamp"] = timestamp_to_ms(time.time())
        logging.info(f"Emitting monitoring event: {event}")
        async with self.emit_event_lock:
            try:
                await self.protocol.emit_monitoring_event(event, queue_event_type)
            except Exception as e:
                logging.error(f"Failed to emit monitoring event: {e}")

    async def run_ingress_loop(self):
        frame_count = 0
        start_time = 0.0
        async for av_frame in self.protocol.ingress_loop(self.stop_event):
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

            # Scale image to 512x512 as most models expect this size, especially when using tensorrt
            width, height = frame.size
            if (width, height) != (512, 512):
                frame_array = np.array(frame)

                # Crop to the center square if image not already square
                square_size = min(width, height)
                if width != height:
                    start_x = width // 2 - square_size // 2
                    start_y = height // 2 - square_size // 2
                    frame_array = frame_array[
                        start_y : start_y + square_size, start_x : start_x + square_size
                    ]

                # Resize using cv2 (much faster than PIL)
                if square_size != 512:
                    frame_array = cv2.resize(frame_array, (512, 512))

                frame = Image.fromarray(frame_array)

            logging.debug(
                f"Sending input frame. Scaled from {width}x{height} to {frame.size[0]}x{frame.size[1]}"
            )
            av_frame = av_frame.replace_image(frame)
            self.process.send_input(av_frame)

            # Increment frame count and measure FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= fps_log_interval:
                status = self.process.get_status()
                logging.info(f"Input FPS: {status.input_status.fps:.2f}")
                frame_count = 0
                start_time = time.time()
        logging.info("Ingress loop ended")

    async def run_egress_loop(self):
        request_id = self.request_id
        async def gen_output_frames() -> AsyncGenerator[OutputFrame, None]:
            frame_count = 0
            start_time = 0.0
            while not self.stop_event.is_set():
                output = await self.process.recv_output()
                if not output:
                    continue

                # TODO accounting for audio output
                if isinstance(output, AudioOutput):
                    logging.debug(
                        f"Output audio received outputRequestId={output.request_id} numFrames={len(output.frames)}"
                    )
                    if output.request_id != request_id:
                        logging.warning(
                            f"Output audio request ID mismatch: expected {request_id}, got {output.request_id}, skipping frame"
                        )
                        continue
                    yield output
                    continue

                if not isinstance(output, VideoOutput):
                    logging.warning(
                        f"Unknown output frame type {type(output)}, dropping"
                    )
                    continue
                if output.request_id != request_id:
                    logging.warning(
                        f"Output video request ID mismatch: expected {request_id}, got {output.request_id}, dropping frame"
                    )
                    continue
                logging.debug(
                    f"Output image received outputRequestId={output.request_id} ts={output.timestamp} time_base={output.time_base} resolution={output.image.width}x{output.image.height} mode={output.image.mode}"
                )


                if not start_time:
                    # only start measuring output FPS after the first frame
                    start_time = time.time()

                yield output

                # Increment frame count and measure FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= fps_log_interval:
                    status = self.process.get_status()
                    logging.info(f"Output FPS: {status.inference_status.fps:.2f}")
                    frame_count = 0
                    start_time = time.time()

        await self.protocol.egress_loop(gen_output_frames())
        logging.info("Egress loop ended")

    async def run_control_loop(self):
        """Consumes control messages from the protocol and updates parameters"""
        async for params in self.protocol.control_loop(self.stop_event):
            try:
                await self.process.update_params(params)
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

import os
import asyncio
import logging
import hashlib
import json
import torch.multiprocessing as mp
import queue
import sys
import time
from typing import Any
import torch

from pipelines import load_pipeline, Pipeline
from log import config_logging, config_logging_fields, log_timing
from trickle import InputFrame, AudioFrame, VideoFrame, OutputFrame, VideoOutput, AudioOutput

class PipelineProcess:
    @staticmethod
    def start(pipeline_name: str, params: dict):
        instance = PipelineProcess(pipeline_name)
        if params:
            instance.update_params(params)
        instance.process.start()
        instance.start_time = time.time()
        return instance

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.ctx = mp.get_context("spawn")

        self.input_queue = self.ctx.Queue(maxsize=1)
        self.output_queue = self.ctx.Queue(maxsize=1)
        self.param_update_queue = self.ctx.Queue()
        self.error_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue(maxsize=100)  # Keep last 100 log lines

        self.pipeline_initialized = self.ctx.Event()
        self.done = self.ctx.Event()
        self.process = self.ctx.Process(target=self.process_loop, args=())
        self.start_time = 0.0
        self.request_id = ""

    def is_alive(self):
        return self.process.is_alive()

    async def stop(self):
        self.done.set()

        if not self.process.is_alive():
            logging.info("Process already not alive")
            return

        logging.info("Terminating pipeline process")

        async def wait_stop(timeout: float) -> bool:
            try:
                await asyncio.to_thread(self.process.join, timeout=timeout)
                return not self.process.is_alive()
            except Exception as e:
                logging.error(f"Process join error: {e}")
                return False

        if not await wait_stop(10):
            logging.error("Failed to terminate process, killing")
            self.process.kill()
            if not await wait_stop(5):
                logging.error("Failed to kill process")

        logging.info("Pipeline process terminated, closing queues")

        for q in [self.input_queue, self.output_queue, self.param_update_queue,
                  self.error_queue, self.log_queue]:
            q.cancel_join_thread()
            q.close()

        logging.info("Pipeline process cleanup complete")

    def is_done(self):
        return self.done.is_set()

    def is_pipeline_initialized(self):
        return self.pipeline_initialized.is_set()

    def update_params(self, params: dict):
        self.param_update_queue.put(params)

    def reset_stream(self, request_id: str, manifest_id: str, stream_id: str):
        # we cannot clear the input_queue as we send CUDA tensors on it, which can't be received by the same process that sent it.
        # So we clear only the other queues, and rely on the request_id checks to avoid using frames from previous sessions.
        clear_queue(self.output_queue)
        clear_queue(self.param_update_queue)
        clear_queue(self.error_queue)
        clear_queue(self.log_queue)
        self.param_update_queue.put({"request_id": request_id, "manifest_id": manifest_id, "stream_id": stream_id})

    # TODO: Once audio is implemented, combined send_input with input_loop
    # We don't need additional queueing as comfystream already maintains a queue
    def send_input(self, frame: InputFrame):
        if isinstance(frame, VideoFrame) and not frame.tensor.is_cuda and torch.cuda.is_available():
            frame = frame.replace_tensor(frame.tensor.cuda())
        self._try_queue_put(self.input_queue, frame)

    async def recv_output(self) -> OutputFrame | None:
        # we cannot do a long get with timeout as that would block the asyncio
        # event loop, so we loop with nowait and sleep async instead.
        # TODO: use asyncio.to_thread instead
        while not self.is_done():
            try:
                return await asyncio.to_thread(self.output_queue.get, timeout=0.1)
            except queue.Empty:
                # Timeout ensures the non-daemon threads from to_thread can exit if task is cancelled
                continue
        return None

    def get_recent_logs(self, n=None) -> list[str]:
        """Get recent logs from the subprocess. If n is None, get all available logs."""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs[-n:] if n is not None else logs  # Only limit if n is specified

    def process_loop(self):
        self._setup_logging()
        pipeline = None

        # Ensure CUDA environment is available inside the subprocess.
        # Multiprocessing (spawn mode) does not inherit environment variables by default,
        # causing `torch.cuda.current_device()` checks in ComfyUI's model_management.py to fail.
        # Explicitly setting `CUDA_VISIBLE_DEVICES` ensures the spawned process recognizes the GPU.
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

        # ComfystreamClient/embeddedComfyClient is not respecting config parameters
        # such as verbose='WARNING', logging_level='WARNING'
        # Setting here to override and supress excessive INFO logging
        # ( load_gpu_models is calling logging.info() for every frame )
        logging.getLogger("comfy").setLevel(logging.WARNING)

        try:
            asyncio.run(self._run_pipeline_loops())
        except Exception as e:
            self._report_error(f"Error in process run method: {e}")


    def _handle_logging_params(self, params: dict) -> bool:
        if isinstance(params, dict) and "request_id" in params and "manifest_id" in params and "stream_id" in params:
            logging.info(f"PipelineProcess: Resetting logging fields with request_id={params['request_id']}, manifest_id={params['manifest_id']} stream_id={params['stream_id']}")
            self.request_id = params["request_id"]
            self._reset_logging_fields(
                params["request_id"], params["manifest_id"], params["stream_id"]
            )
            return True
        return False

    async def _initialize_pipeline(self):
        try:
            params = await self._get_latest_params(timeout=0.005)
            if params is not None:
                logging.info(f"PipelineProcess: Got params from param_update_queue {params}")
            else:
                logging.info("PipelineProcess: No params found in param_update_queue, loading with default params")
                params = {}

            with log_timing(f"PipelineProcess: Pipeline loading with {params}"):
                pipeline = load_pipeline(self.pipeline_name)
                await pipeline.initialize(**params)
                return pipeline
        except Exception as e:
            self._report_error(f"Error loading pipeline: {e}")
            if not params:
                # Already tried loading with default params
                raise
            try:
                with log_timing(
                    f"PipelineProcess: Pipeline loading with default params due to error with params: {params}"
                ):
                    pipeline = load_pipeline(self.pipeline_name)
                    await pipeline.initialize()
                    return pipeline
            except Exception as e:
                self._report_error(f"Error loading pipeline with default params: {e}")
                raise

    async def _run_pipeline_loops(self):
        pipeline = await self._initialize_pipeline()
        self.pipeline_initialized.set()
        input_task = asyncio.create_task(self._input_loop(pipeline))
        output_task = asyncio.create_task(self._output_loop(pipeline))
        param_task = asyncio.create_task(self._param_update_loop(pipeline))

        async def wait_for_stop():
            while not self.is_done():
                await asyncio.sleep(0.1)

        tasks = [input_task, output_task, param_task, asyncio.create_task(wait_for_stop())]

        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            self._report_error(f"Error in pipeline loops: {e}")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._cleanup_pipeline(pipeline)

        logging.info("PipelineProcess: _run_pipeline_loops finished.")

    async def _input_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                input_frame = await asyncio.to_thread(self.input_queue.get, timeout=0.1)
                if isinstance(input_frame, VideoFrame):
                    input_frame.log_timestamps["pre_process_frame"] = time.time()
                    await pipeline.put_video_frame(input_frame, self.request_id)
                elif isinstance(input_frame, AudioFrame):
                    self._try_queue_put(self.output_queue, AudioOutput([input_frame], self.request_id))
            except queue.Empty:
                # Timeout ensures the non-daemon threads from to_thread can exit if task is cancelled
                continue
            except Exception as e:
                self._report_error(f"Error processing input frame: {e}")

    async def _output_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                output = await pipeline.get_processed_video_frame()
                if isinstance(output, VideoOutput) and not output.tensor.is_cuda and torch.cuda.is_available():
                    output = output.replace_tensor(output.tensor.cuda())
                output.log_timestamps["post_process_frame"] = time.time()
                self._try_queue_put(self.output_queue, output)
            except Exception as e:
                self._report_error(f"Error processing output frame: {e}")

    async def _param_update_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                params = await self._get_latest_params(timeout=0.1)
                if params is None:
                    continue

                params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
                logging.info(f"PipelineProcess: Updating pipeline parameters: hash={params_hash} params={params}")

                with log_timing(f"PipelineProcess: Pipeline update parameters with params_hash={params_hash}"):
                    await pipeline.update_params(**params)
            except Exception as e:
                self._report_error(f"Error updating params: {e}")

    async def _get_latest_params(self, timeout: float) -> dict | None:
        """
        Get the latest params from the param_update_queue, skipping stale entries before the latest. Already filters
        and processes params that are only logging updates. Waits for timeout seconds for a new params entry, or returns
        None if no new entry is found.
        """

        try:
            params = await asyncio.to_thread(self.param_update_queue.get, timeout=timeout)
        except queue.Empty:
            return None

        if self._handle_logging_params(params):
            params = None

        # Drain the params queue to get the latest params
        while not self.param_update_queue.empty():
            try:
                new_params = self.param_update_queue.get_nowait()
                if not self._handle_logging_params(new_params):
                    params = new_params
            except queue.Empty:
                break

        return params

    def _report_error(self, error_msg: str):
        error_event = {
            "message": error_msg,
            "timestamp": time.time()
        }
        logging.error(error_msg)
        self._try_queue_put(self.error_queue, error_event)

    async def _cleanup_pipeline(self, pipeline):
        if pipeline is not None:
            try:
                await pipeline.stop()
            except Exception as e:
                logging.error(f"Error stopping pipeline: {e}")

    def _setup_logging(self):
        level = (
            logging.DEBUG if os.environ.get("VERBOSE_LOGGING") == "1" else logging.INFO
        )
        logger = config_logging(log_level=level)
        queue_handler = LogQueueHandler(self)
        config_logging_fields(queue_handler, "", "", "")
        logger.addHandler(queue_handler)

        self.queue_handler = queue_handler

        # Tee stdout and stderr to our log queue while preserving original output
        sys.stdout = QueueTeeStream(sys.stdout, self)
        sys.stderr = QueueTeeStream(sys.stderr, self)

    def _reset_logging_fields(self, request_id: str, manifest_id: str, stream_id: str):
        config_logging(request_id=request_id, manifest_id=manifest_id, stream_id=stream_id)
        config_logging_fields(self.queue_handler, request_id, manifest_id, stream_id)

    def _try_queue_put(self, _queue: mp.Queue, item: Any):
        """Helper to put an item on a queue, only if there's room"""
        try:
            _queue.put_nowait(item)
        except queue.Full:
            pass

    def get_last_error(self) -> tuple[str, float] | None:
        """Get the most recent error and its timestamp from the error queue, if any"""
        last_error = None
        while True:
            try:
                last_error = self.error_queue.get_nowait()
            except queue.Empty:
                break
        return (last_error["message"], last_error["timestamp"]) if last_error else None

class QueueTeeStream:
    """Tee all stream (stdout or stderr) messages to the process log queue"""
    def __init__(self, original_stream, process: PipelineProcess):
        self.original_stream = original_stream
        self.process = process

    def write(self, text):
        self.original_stream.write(text)
        text = text.strip()  # Only queue non-empty lines
        if text:
            self.process._try_queue_put(self.process.log_queue, text)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return self.original_stream.isatty()

class LogQueueHandler(logging.Handler):
    """Send all log records to the process's log queue"""
    def __init__(self, process: PipelineProcess):
        super().__init__()
        self.process = process

    def emit(self, record):
        msg = self.format(record)
        self.process._try_queue_put(self.process.log_queue, msg)

# Function to clear the queue
def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()  # Remove items without blocking
        except Exception as e:
            logging.error(f"Error while clearing queue: {e}")

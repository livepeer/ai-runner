import os
import asyncio
import logging
import hashlib
import json
import torch.multiprocessing as mp
import queue
import sys
import signal
import threading
import time
from typing import Any

import torch

from ..pipelines import load_pipeline, Pipeline, BaseParams
from ..log import config_logging, config_logging_fields, log_timing
from ..trickle import (
    InputFrame,
    AudioFrame,
    VideoFrame,
    OutputFrame,
    VideoOutput,
    AudioOutput,
)

from .loading_overlay import LoadingOverlayRenderer


class PipelineProcess:
    @staticmethod
    def start(pipeline_name: str, params: dict | None = None):
        instance = PipelineProcess(pipeline_name)
        if params is not None:
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

        self.pipeline_ready = self.ctx.Event()
        self.pipeline_ready_time = self.ctx.Value("d", 0.0)
        self.done = self.ctx.Event()
        self.process = self.ctx.Process(target=self.process_loop, args=())
        self.start_time = 0.0
        self.request_id = ""

        # Using underscored names to emphasize state used only from the child process.
        self._last_params = BaseParams()

    def is_alive(self):
        return self.process.is_alive()

    async def stop(self):
        self.done.set()

        is_terminating = False
        if not self.is_alive():
            logging.info("Process already not alive")
        else:
            logging.info("Terminating pipeline process")
            is_terminating = True
            self.process.terminate()

            if await self._wait_stop(5):
                is_terminating = False

        logging.info("Closing process queues")
        for q in [
            self.input_queue,
            self.output_queue,
            self.param_update_queue,
            self.error_queue,
            self.log_queue,
        ]:
            q.cancel_join_thread()
            q.close()

        if is_terminating and self.is_alive():
            logging.error("Failed to terminate process, killing")
            self.process.kill()
            if not await self._wait_stop(3):
                logging.error(
                    f"Failed to kill process self_pid={os.getpid()} child_pid={self.process.pid} is_alive={self.process.is_alive()}"
                )
                raise RuntimeError("Failed to kill process")

        logging.info("Pipeline process cleanup complete")

    async def _wait_stop(self, timeout: float) -> bool:
        """
        Wait for the process to stop and return True if it did, False otherwise.
        """
        try:
            await asyncio.to_thread(self.process.join, timeout=timeout)
            return not self.process.is_alive()
        except Exception as e:
            logging.error(f"Process join error: {e}")
            return False

    def is_done(self):
        return self.done.is_set()

    def is_pipeline_ready(self) -> tuple[bool, float | None]:
        """
        Returns a tuple [bool, float] where the bool indicates if the pipeline is
        ready and the float is the timestamp of when it became ready.
        """
        # Also return not ready if the process is shutting down (done event is set)
        if not self.pipeline_ready.is_set() or self.done.is_set():
            return (False, None)

        with self.pipeline_ready_time.get_lock():
            return (True, self.pipeline_ready_time.value)

    def _is_loading(self) -> bool:
        is_ready, _ = self.is_pipeline_ready()
        return not is_ready

    def _set_pipeline_ready(self, ready: bool):
        if not ready:
            self.pipeline_ready.clear()
            return

        with self.pipeline_ready_time.get_lock():
            self.pipeline_ready_time.value = time.time()
        self.pipeline_ready.set()

    def update_params(self, params: dict):
        self.param_update_queue.put(params)

    def reset_stream(self, request_id: str, manifest_id: str, stream_id: str):
        # we cannot clear the input_queue as we send CUDA tensors on it, which can't be received by the same process that sent it.
        # So we clear only the other queues, and rely on the request_id checks to avoid using frames from previous sessions.
        clear_queue(self.output_queue)
        clear_queue(self.param_update_queue)
        clear_queue(self.error_queue)
        clear_queue(self.log_queue)
        self.param_update_queue.put(
            {
                "request_id": request_id,
                "manifest_id": manifest_id,
                "stream_id": stream_id,
            }
        )

    # TODO: Once audio is implemented, combined send_input with input_loop
    # We don't need additional queueing as comfystream already maintains a queue
    def send_input(self, frame: InputFrame):
        if isinstance(frame, VideoFrame) and not frame.tensor.is_cuda and torch.cuda.is_available():
            frame = frame.replace_tensor(frame.tensor.cuda())
        self._try_queue_put(self.input_queue, frame)

    async def recv_output(self) -> OutputFrame | None:
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
        _setup_signal_handlers(self.done)
        _setup_parent_death_signal()
        _start_parent_watchdog(self.done)
        self._setup_logging()

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
            self._report_error("Error in process run method", e)

    def _handle_logging_params(self, params: dict) -> bool:
        if isinstance(params, dict) and "request_id" in params and "manifest_id" in params and "stream_id" in params:
            logging.info(
                f"PipelineProcess: Resetting logging fields with request_id={params['request_id']}, manifest_id={params['manifest_id']} stream_id={params['stream_id']}"
            )
            self.request_id = params["request_id"]
            self._reset_logging_fields(params["request_id"], params["manifest_id"], params["stream_id"])
            return True
        return False

    async def _initialize_pipeline(self):
        try:
            params = await self._get_latest_params(timeout=0.1)
            if params is not None:
                logging.info(f"PipelineProcess: Got params from param_update_queue {params}")
            else:
                logging.info("PipelineProcess: No params found in param_update_queue, loading with default params")
                params = {}

            with log_timing(f"PipelineProcess: Pipeline loading with {params}"):
                self._last_params = BaseParams(**params)
                pipeline = load_pipeline(self.pipeline_name)
                await pipeline.initialize(**params)
                return pipeline
        except Exception as e:
            self._report_error("Error loading pipeline", e)
            if not params:
                # Already tried loading with default params
                raise
            try:
                with log_timing(
                    f"PipelineProcess: Pipeline loading with default params due to error with params: {params}"
                ):
                    self._last_params = BaseParams()
                    pipeline = load_pipeline(self.pipeline_name)
                    await pipeline.initialize()
                    return pipeline
            except Exception as e:
                self._report_error("Error loading pipeline with default params", e)
                raise

    async def _run_pipeline_loops(self):
        overlay = LoadingOverlayRenderer()
        pipeline = await self._initialize_pipeline()
        input_task = asyncio.create_task(self._input_loop(pipeline, overlay))
        output_task = asyncio.create_task(self._output_loop(pipeline, overlay))
        param_task = asyncio.create_task(self._param_update_loop(pipeline, overlay))
        self._set_pipeline_ready(True)

        async def wait_for_stop():
            while not self.is_done():
                await asyncio.sleep(0.1)

        tasks = [input_task, output_task, param_task, asyncio.create_task(wait_for_stop())]

        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception as e:
            self._report_error("Error in pipeline loops", e)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._cleanup_pipeline(pipeline, overlay)

        logging.info("PipelineProcess: _run_pipeline_loops finished.")

    async def _input_loop(self, pipeline: Pipeline, overlay: LoadingOverlayRenderer):
        while not self.is_done():
            try:
                input = await asyncio.to_thread(self.input_queue.get, timeout=0.1)
                if isinstance(input, VideoFrame):
                    input.log_timestamps["pre_process_frame"] = time.time()

                    if self._is_loading() and self._last_params.show_reloading_frame:
                        await self._render_loading_frame(overlay, input)
                    else:
                        await pipeline.put_video_frame(input, self.request_id)
                elif isinstance(input, AudioFrame):
                    self._try_queue_put(self.output_queue, AudioOutput([input], self.request_id))
            except queue.Empty:
                # Timeout ensures the non-daemon threads from to_thread can exit if task is cancelled
                continue
            except Exception as e:
                self._report_error("Error processing input frame", e)

    async def _render_loading_frame(self, overlay: LoadingOverlayRenderer, input: VideoFrame):
        if not overlay.is_active():
            overlay.begin_reload()

        w, h = self._last_params.width, self._last_params.height
        loading_tensor = await overlay.render(w, h)
        if torch.cuda.is_available() and not loading_tensor.is_cuda:
            loading_tensor = loading_tensor.cuda()

        out_frame = input.replace_tensor(loading_tensor)
        out = VideoOutput(out_frame, self.request_id, is_loading_frame=True)

        out.log_timestamps["post_process_frame"] = time.time()
        self._try_queue_put(self.output_queue, out)

    async def _output_loop(self, pipeline: Pipeline, overlay: LoadingOverlayRenderer):
        while not self.is_done():
            try:
                out = await pipeline.get_processed_video_frame()
                overlay.update_last_frame(out.tensor)

                if overlay.is_active():
                    if self._is_loading():
                        # Ignore frames that may come after the loading started to avoid thrashing the loading overlay.
                        continue
                    overlay.end_reload()

                if isinstance(out, VideoOutput) and not out.tensor.is_cuda and torch.cuda.is_available():
                    out = out.replace_tensor(out.tensor.cuda())

                out.log_timestamps["post_process_frame"] = time.time()
                self._try_queue_put(self.output_queue, out)
            except Exception as e:
                self._report_error("Error processing output frame", e)

    async def _param_update_loop(self, pipeline: Pipeline, overlay: LoadingOverlayRenderer):
        while not self.is_done():
            reload_task = None
            try:
                params = await self._get_latest_params(timeout=0.1)
                if params is None:
                    continue

                params_hash = hashlib.md5(
                    json.dumps(params, sort_keys=True).encode()
                ).hexdigest()
                logging.info(
                    f"PipelineProcess: Updating pipeline parameters: hash={params_hash} params={params}"
                )

                with log_timing(f"PipelineProcess: Pipeline update parameters with params_hash={params_hash}"):
                    reload_task = await pipeline.update_params(**params)
                    self._last_params = BaseParams(**params)
            except Exception as e:
                self._report_error("Error updating params", e)
                continue

            try:
                if reload_task is None:
                    # This means update_params was already able to update the pipeline dynamically
                    continue
                with log_timing("PipelineProcess: Reloading pipeline"):
                    self._set_pipeline_ready(False)
                    await reload_task
                    self._set_pipeline_ready(True)
            except Exception as e:
                # Reloading pipeline failed so we have to exit the process so it's restarted from scratch.
                self._report_error("Error reloading pipeline", e)
                self.done.set()
                # Schedule a delayed exit to make sure the process doesn't hang
                threading.Thread(target=lambda: (time.sleep(3), os._exit(1)), daemon=True).start()
            finally:
                # Pre-warm the loading overlay, so it's shown with the new resolution on the next reload.
                try:
                    await overlay.prewarm(self._last_params.width, self._last_params.height)
                except Exception:
                    logging.warning("Failed to prewarm loading overlay caches", exc_info=True)

    async def _get_latest_params(self, timeout: float) -> dict | None:
        """
        Get the latest params from the param_update_queue, skipping stale entries before the latest. Already filters
        and processes params that are only logging updates. Waits for timeout seconds for a new params entry, or returns
        None if no new entry is found.
        """

        try:
            params = await asyncio.to_thread(
                self.param_update_queue.get, timeout=timeout
            )
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

    def _report_error(self, msg: str, error: Exception | None = None, silent=False):
        if not silent:
            logging.error(msg, exc_info=error)

        error_event = {
            "message": f"{msg}: {error}" if error else msg,
            "timestamp": time.time(),
        }
        self._try_queue_put(self.error_queue, error_event)

    async def _cleanup_pipeline(self, pipeline: Pipeline, overlay: LoadingOverlayRenderer):
        overlay.end_reload()
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
        config_logging(
            request_id=request_id, manifest_id=manifest_id, stream_id=stream_id
        )
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
        try:
            if getattr(record, "report_error", False):
                self.process._report_error(record.getMessage(), silent=True)
        except Exception as e:
            logging.error(f"Error reporting error: {e}")


# Function to clear the queue
def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()  # Remove items without blocking
        except Exception as e:
            logging.error(f"Error while clearing queue: {e}")


def _setup_signal_handlers(
    done: mp.Event,
    signals: list[signal.Signals] = [signal.SIGTERM, signal.SIGINT],
):
    """
    Install signal handlers for graceful shutdown in the process. When a signal is received,
    we set the provided done_event (supports multiprocessing.Event or similar interfaces).
    """

    def _handle(sig, _frame):
        logging.info(f"Received signal: {sig}. Initiating graceful shutdown.")
        try:
            done.set()
        except Exception as e:
            logging.error(
                "Terminating process (child) due to failure handling signal",
                exc_info=e,
            )
            os._exit(1)

    for sig in signals:
        signal.signal(sig, _handle)


is_linux = sys.platform.startswith("linux")
is_unix = is_linux or sys.platform == "darwin"


def _setup_parent_death_signal():
    """
    Ensure the child gets a SIGTERM if the parent dies when running in Linux.
    This is a best-effort attempt, and errors are logged but ignored.
    """
    if not is_linux:
        logging.info(
            f"Skipping Linux-only parent death signal setup due to unsupported platform={sys.platform}"
        )
        return

    try:
        import ctypes
        from ctypes.util import find_library

        libc_path = find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_path, use_errno=True)

        # This is the code for the "parent death signal" feature in Linux
        PR_SET_PDEATHSIG = 1
        res = libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        if res != 0:
            err = ctypes.get_errno()
            logging.warning(f"prctl(PR_SET_PDEATHSIG) failed with errno={err}")
    except Exception as e:
        logging.warning(f"Unable to set PDEATHSIG: {e}")


def _start_parent_watchdog(done: mp.Event):
    """
    Start a lightweight watchdog to observe parent death as a cross-platform fallback
    """

    # Only supported on Unix-like systems where PPID becomes 1 after parent death.
    # TODO: Add Windows support using a parent process handle wait (OpenProcess + WaitForSingleObject).
    if not is_unix:
        logging.info(
            f"Skipping Unix-only parent watchdog due to unsupported platform={sys.platform}"
        )
        return

    def _watch_parent():
        try:
            while not done.is_set():
                time.sleep(1)
                if os.getppid() == 1:
                    logging.error(
                        "Parent process died; initiating graceful shutdown in child"
                    )
                    done.set()
                    break
        except Exception as e:
            logging.error(
                "Terminating child process due to failure in watchdog",
                exc_info=e,
            )
            os._exit(1)

    t = threading.Thread(target=_watch_parent, name="parent-watchdog", daemon=True)
    t.start()

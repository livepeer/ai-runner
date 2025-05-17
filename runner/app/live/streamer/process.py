import os
import asyncio
import logging
import multiprocessing as mp
import queue
import sys
import threading
import time
from typing import Any, Union
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

        self.input_queue = self.ctx.Queue(maxsize=5)
        self.output_queue = self.ctx.Queue()
        self.param_update_queue = self.ctx.Queue()
        self.error_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue(maxsize=100)  # Re-enable log_queue
        # self.log_queue = None # DIAGNOSTIC: Disable log_queue entirely (REVERTED)

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

        # NEW: Close parent's queue handles first
        logging.info("Parent process: Closing its queue handles BEFORE attempting to join child.")
        # Using a list of tuples for names and instances for clearer logging
        queues_to_close_parent_side = [
            ("input_queue", self.input_queue),
            ("output_queue", self.output_queue),
            ("param_update_queue", self.param_update_queue),
            ("error_queue", self.error_queue),
            ("log_queue", self.log_queue) # self.log_queue might be None if diagnostic was re-enabled
        ]
        for q_name, q_instance in queues_to_close_parent_side:
            if q_instance: # Check if the queue instance exists
                try:
                    # These are the parent's handles to the queues.
                    q_instance.cancel_join_thread() # Signal feeder thread associated with this handle to stop.
                    q_instance.close()              # Close this end of the queue.
                    logging.info(f"Parent process: Closed its handle for {q_name}.")
                except Exception as e_parent_q_close:
                    logging.error(f"Parent process: Error closing its handle for {q_name}: {e_parent_q_close}")
            else:
                logging.warning(f"Parent process: Queue {q_name} instance is None, cannot close handle.")
        logging.info("Parent process: Finished closing its queue handles.")

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

        logging.info("Terminated pipeline process")

        # OLD Location of queue cleanup - MOVED EARLIER
        # for q in [self.input_queue, self.output_queue, self.param_update_queue,
        #           self.error_queue, self.log_queue]:
        #     q.cancel_join_thread()
        #     q.close()
        # logging.info("All queues closed and join threads cancelled.")

    def is_done(self):
        return self.done.is_set()

    def is_pipeline_initialized(self):
        return self.pipeline_initialized.is_set()

    def update_params(self, params: dict):
        self.param_update_queue.put(params)

    def reset_stream(self, request_id: str, stream_id: str):
        clear_queue(self.input_queue)
        clear_queue(self.output_queue)
        clear_queue(self.param_update_queue)
        clear_queue(self.error_queue)
        clear_queue(self.log_queue)
        self.param_update_queue.put({"request_id": request_id, "stream_id": stream_id})

    # TODO: Once audio is implemented, combined send_input with input_loop
    # We don't need additional queueing as comfystream already maintains a queue
    def send_input(self, frame: InputFrame):
        self._queue_put_fifo(self.input_queue, frame)

    async def recv_output(self) -> OutputFrame | None:
        # we cannot do a long get with timeout as that would block the asyncio
        # event loop, so we loop with nowait and sleep async instead.
        # TODO: use asyncio.to_thread instead
        while not self.is_done():
            try:
                # Check if output_queue exists and is not None, though it should always exist here
                if self.output_queue:
                    output = self.output_queue.get_nowait()
                    return output
                else:
                    logging.warning("PipelineProcess: recv_output - self.output_queue is None, cannot get.")
                    await asyncio.sleep(0.05) # Avoid busy loop if queue is unexpectedly None
                    return None # Or handle as an error state
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue
        return None

    def get_recent_logs(self, n=None) -> list[str]:
        """Get recent logs from the subprocess. If n is None, get all available logs."""
        if not self.log_queue: # DIAGNOSTIC: Check if log_queue is None
            return []

        logs = []
        # Defensive check: ensure log_queue is still not None before calling methods on it.
        if self.log_queue:
            while not self.log_queue.empty():
                try:
                    logs.append(self.log_queue.get_nowait())
                except queue.Empty:
                    break
        return logs[-n:] if n is not None else logs

    def process_loop(self):
        undo_logging_setup_fn = self._setup_logging()
        pipeline = None
        logging.info("PipelineProcess: process_loop started.")

        # Ensure CUDA environment is available inside the subprocess.
        # Multiprocessing (spawn mode) does not inherit environment variables by default,
        # causing `torch.cuda.current_device()` checks in ComfyUI's model_management.py to fail.
        # Explicitly setting `CUDA_VISIBLE_DEVICES` ensures the spawned process recognizes the GPU.
        logging.info("PipelineProcess: process_loop - Before PyTorch CUDA check.")
        if torch.cuda.is_available():
            logging.info("PipelineProcess: process_loop - PyTorch CUDA is available. Setting CUDA_VISIBLE_DEVICES.")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())
            logging.info(f"PipelineProcess: process_loop - CUDA_VISIBLE_DEVICES set to {os.environ.get('CUDA_VISIBLE_DEVICES')}.")
        else:
            logging.info("PipelineProcess: process_loop - PyTorch CUDA is NOT available.")
        logging.info("PipelineProcess: process_loop - After PyTorch CUDA check.")

        # ComfystreamClient/embeddedComfyClient is not respecting config parameters
        # such as verbose='WARNING', logging_level='WARNING'
        # Setting here to override and supress excessive INFO logging
        # ( load_gpu_models is calling logging.info() for every frame )
        logging.getLogger("comfy").setLevel(logging.WARNING)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            logging.info("PipelineProcess: process_loop - About to call loop.run_until_complete with _run_pipeline_loops.")
            loop.run_until_complete(self._run_pipeline_loops())
            logging.info("PipelineProcess: process_loop - loop.run_until_complete(_run_pipeline_loops) completed.")
        except Exception as e:
            self._report_error(f"Error in process_loop (during asyncio execution): {e}")
        finally:
            logging.info("PipelineProcess: process_loop - Ensuring all asyncio tasks on loop are handled before closing loop.")
            try:
                all_tasks = asyncio.all_tasks(loop=loop)
                # Exclude the task that might be running this finally block if run_until_complete was used with a gather
                # current_task = asyncio.current_task(loop=loop) # Can be None if not in a task
                # tasks_to_finalize = {task for task in all_tasks if task is not current_task}
                tasks_to_finalize = all_tasks

                if tasks_to_finalize:
                    logging.info(f"PipelineProcess: process_loop - Found {len(tasks_to_finalize)} outstanding asyncio tasks. Cancelling and gathering.")
                    for task in tasks_to_finalize:
                        if not task.done():
                            task.cancel()
                    # Run gather on the loop to allow cancellations to be processed.
                    loop.run_until_complete(asyncio.gather(*tasks_to_finalize, return_exceptions=True))
                    logging.info("PipelineProcess: process_loop - Outstanding asyncio tasks gathered.")
                else:
                    logging.info("PipelineProcess: process_loop - No outstanding asyncio tasks found on loop before close.")
            except Exception as e_task_cleanup:
                 # Use logging here as log_queue should still be operational if _undo_logging_setup hasn't run
                logging.error(f"PipelineProcess: process_loop - Error during outstanding task cleanup before loop close: {e_task_cleanup}", exc_info=True)

            logging.info("PipelineProcess: process_loop - Closing asyncio event loop.")
            loop.close()
            logging.info("PipelineProcess: process_loop - Asyncio event loop closed.")

            # Final cleanup of logging and listing threads
            try:
                logging.info("PipelineProcess: process_loop - About to list active threads.")
                active_threads = threading.enumerate()
                thread_details = []
                for t in active_threads:
                    thread_details.append(f"Name='{t.name}', Daemon={t.daemon}, Alive={t.is_alive()}")
                logging.info(f"PipelineProcess: process_loop - Active threads: {thread_details}")

                logging.info("PipelineProcess: process_loop - Reached final cleanup block. Attempting to clean up logging resources.")
                if undo_logging_setup_fn:
                    undo_logging_setup_fn()
                    current_out_stream = getattr(sys, 'stdout', sys.__stdout__)
                    print(f"PipelineProcess: process_loop - undo_logging_setup_fn executed. Original stdout/stderr restored.", file=current_out_stream)
                else:
                    current_out_stream = getattr(sys, 'stdout', sys.__stdout__)
                    print("PipelineProcess: process_loop - undo_logging_setup_fn was not defined, skipping call.", file=current_out_stream)

            except Exception as e_final_cleanup:
                # This logging.error might be affected if os.close(1) or os.close(2) succeeded and logging depended on them via LogQueueHandler's underlying pipe.
                logging.error(f"PipelineProcess: process_loop - CRITICAL: Exception during final cleanup/fd closing/thread listing: {e_final_cleanup}", exc_info=True)
                # As a last resort, print to sys.__stderr__ if it hasn't been closed or is still valid somehow.
                fallback_err_stream = getattr(sys, 'stderr', sys.__stderr__)
                print(f"PipelineProcess: process_loop - CRITICAL: Exception during final cleanup/fd closing/thread listing: {e_final_cleanup}", file=fallback_err_stream)
            finally:
                # Check for active child processes from multiprocessing module
                # This log will go to the original/restored stdout.
                final_print_stream = getattr(sys, 'stdout', sys.__stdout__)
                try:
                    children = mp.active_children()
                    print(f"PipelineProcess: process_loop - Active children according to multiprocessing: {children}", file=final_print_stream)
                except Exception as e_active_children:
                    print(f"PipelineProcess: process_loop - Error checking for active_children: {e_active_children}", file=final_print_stream)

                print("PipelineProcess: process_loop attempting to exit.", file=final_print_stream)

    def _handle_logging_params(self, params: dict) -> dict:
        if isinstance(params, dict) and "request_id" in params and "stream_id" in params:
            logging.info(f"PipelineProcess: Resetting logging fields with request_id={params['request_id']}, stream_id={params['stream_id']}")
            self.request_id = params["request_id"]
            self._reset_logging_fields(
                params["request_id"], params["stream_id"]
            )
            return {}
        return params

    async def _initialize_pipeline(self):
        try:
            stream_id = ""
            params = {}
            try:
                params = self.param_update_queue.get_nowait()
                logging.info(f"PipelineProcess: Got params from param_update_queue {params}")
                params = self._handle_logging_params(params)
            except queue.Empty:
                logging.info("PipelineProcess: No params found in param_update_queue, loading with default params")

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
        logging.info("PipelineProcess: _run_pipeline_loops started.")
        pipeline = None
        tasks = [] # Define tasks in a broader scope for the finally block
        try:
            pipeline = await self._initialize_pipeline()
            self.pipeline_initialized.set()
            logging.info("PipelineProcess: Pipeline initialized and event set.")

            logging.info("PipelineProcess: Creating input_loop task.")
            input_task = asyncio.create_task(self._input_loop(pipeline), name="input_loop")
            logging.info("PipelineProcess: Creating output_loop task.")
            output_task = asyncio.create_task(self._output_loop(pipeline), name="output_loop")
            logging.info("PipelineProcess: Creating param_update_loop task.")
            param_task = asyncio.create_task(self._param_update_loop(pipeline), name="param_update_loop")

            async def wait_for_stop_helper():
                logging.info("PipelineProcess: wait_for_stop_helper task started.")
                while not self.is_done():
                    await asyncio.sleep(0.1)
                logging.info("PipelineProcess: wait_for_stop_helper task finished (self.done was set).")

            wait_task = asyncio.create_task(wait_for_stop_helper(), name="wait_for_stop_helper")

            tasks = [input_task, output_task, param_task, wait_task]
            task_names_for_log = [t.get_name() for t in tasks if hasattr(t, 'get_name')]
            logging.info(f"PipelineProcess: Waiting for tasks {task_names_for_log} (FIRST_COMPLETED).")

            done_tasks, pending_tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            logging.info(f"PipelineProcess: asyncio.wait completed. Done tasks: {len(done_tasks)}, Pending: {len(pending_tasks)}.")

        except asyncio.CancelledError:
            logging.info("PipelineProcess: _run_pipeline_loops was cancelled.")
            # Tasks will be cancelled in the finally block
            raise
        except Exception as e:
            logging.error(f"PipelineProcess: Exception in _run_pipeline_loops: {e}", exc_info=True)
            self._report_error(f"Error in _run_pipeline_loops: {e}")
        finally:
            logging.info("PipelineProcess: _run_pipeline_loops - Entering finally block for task cleanup.")

            if tasks: # Check if tasks list was initialized
                logging.info("PipelineProcess: _run_pipeline_loops - Cleaning up created tasks.")
                for task in tasks:
                    task_name = task.get_name() if hasattr(task, 'get_name') else "Unnamed_task"
                    if not task.done():
                        logging.info(f"PipelineProcess: _run_pipeline_loops - Cancelling task '{task_name}'.")
                        task.cancel()
                    else:
                        try:
                            exc = task.exception()
                            if exc:
                                logging.info(f"PipelineProcess: _run_pipeline_loops - Task '{task_name}' already done with exception: {exc}")
                        except asyncio.CancelledError:
                            logging.info(f"PipelineProcess: _run_pipeline_loops - Task '{task_name}' already done and was cancelled.")
                        except asyncio.InvalidStateError:
                            pass

                logging.info("PipelineProcess: _run_pipeline_loops - Gathering tasks after cancellation.")
                await asyncio.gather(*tasks, return_exceptions=True)
                # Logging for gathered results can be added here if needed, similar to _simplified_run_loops
                logging.info("PipelineProcess: _run_pipeline_loops - Finished gathering tasks.")

            # Additional sweep for any other tasks on the loop (optional, but good for robustness)
            try:
                current_loop = asyncio.get_running_loop()
                all_loop_tasks = asyncio.all_tasks(loop=current_loop)
                # Exclude the current task itself (which is _run_pipeline_loops if it wasn't awaited directly by run_until_complete)
                # and any tasks already handled.
                # For simplicity, let's assume `tasks` covers what we explicitly manage.
                # A more robust sweep might be needed if libraries spawn unmanaged tasks.
                # For now, we rely on the explicit task cancellation above.
                other_tasks_on_loop = {t for t in all_loop_tasks if t is not asyncio.current_task(loop=current_loop) and t not in tasks}
                if other_tasks_on_loop:
                    logging.info(f"PipelineProcess: _run_pipeline_loops - Found {len(other_tasks_on_loop)} other unmanaged tasks. Cancelling.")
                    for task in other_tasks_on_loop:
                        if not task.done(): task.cancel()
                    await asyncio.gather(*other_tasks_on_loop, return_exceptions=True)
            except RuntimeError: # Can happen if loop is already closed or not running
                logging.warning("PipelineProcess: _run_pipeline_loops - Could not get running loop for additional task sweep, or loop already closed.")
            except Exception as e_extra_sweep:
                logging.error(f"PipelineProcess: _run_pipeline_loops - Error during additional task sweep: {e_extra_sweep}")

            if pipeline:
                logging.info("PipelineProcess: _run_pipeline_loops - Cleaning up pipeline object.")
                await self._cleanup_pipeline(pipeline)

            logging.info("PipelineProcess: _run_pipeline_loops - Cleaning up process queues.")
            self._cleanup_process_queues()
            logging.info("PipelineProcess: _run_pipeline_loops finished.")

    async def _input_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                input_frame = await asyncio.to_thread(self.input_queue.get, timeout=0.1)
                if isinstance(input_frame, VideoFrame):
                    input_frame.log_timestamps["pre_process_frame"] = time.time()
                    await pipeline.put_video_frame(input_frame, self.request_id)
                elif isinstance(input_frame, AudioFrame):
                    self._queue_put_fifo(self.output_queue, AudioOutput([input_frame], self.request_id))
            except queue.Empty:
                continue
            except Exception as e:
                self._report_error(f"Error processing input frame: {e}")

    async def _output_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                output_frame = await pipeline.get_processed_video_frame()
                output_frame.log_timestamps["post_process_frame"] = time.time()
                await asyncio.to_thread(self.output_queue.put, output_frame, timeout=0.1)
            except queue.Full:
                continue
            except Exception as e:
                self._report_error(f"Error processing output frame: {e}")

    async def _param_update_loop(self, pipeline: Pipeline):
        while not self.is_done():
            try:
                params = await asyncio.to_thread(self.param_update_queue.get, timeout=0.1)
                if self._handle_logging_params(params):
                    logging.info(f"PipelineProcess: Updating pipeline parameters: {params}")
                    await pipeline.update_params(**params)
            except queue.Empty:
                continue
            except Exception as e:
                self._report_error(f"Error updating params: {e}")

    def _report_error(self, error_msg: str):
        error_event = {
            "message": error_msg,
            "timestamp": time.time()
        }
        logging.error(error_msg)
        self._queue_put_fifo(self.error_queue, error_event)

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
        logger_instance = config_logging(log_level=level)
        self.logger_instance_ref = logger_instance # Store the logger instance for _undo_logging_setup

        # Re-enable LogQueueHandler and its addition to logger
        self.current_queue_handler = LogQueueHandler(self)
        config_logging_fields(self.current_queue_handler, "", "")
        logger_instance.addHandler(self.current_queue_handler)
        # logging.info("PipelineProcess: _setup_logging - DIAGNOSTIC: LogQueueHandler creation and logger.addHandler SKIPPED.") # REVERTED

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Re-enable stdout/stderr redirection
        sys.stdout = QueueTeeStream(original_stdout, self)
        sys.stderr = QueueTeeStream(original_stderr, self)
        # logging.info("PipelineProcess: _setup_logging - DIAGNOSTIC: sys.stdout and sys.stderr redirection to QueueTeeStream is DISABLED.") # REVERTED

        def _undo_logging_setup():
            # logging.info("PipelineProcess: _undo_logging_setup - Starting cleanup.") # This log won't go to parent if handler disabled

            try:
                if hasattr(self, 'current_queue_handler') and self.current_queue_handler and hasattr(self, 'logger_instance_ref') and self.logger_instance_ref:
                    self.logger_instance_ref.removeHandler(self.current_queue_handler)
                    self.current_queue_handler.close()
                    self.current_queue_handler = None # Clear it from self after cleanup
                    # logging.info("PipelineProcess: _undo_logging_setup - LogQueueHandler removed and closed.") # Won't go to parent
                # else:
                    # logging.info("PipelineProcess: _undo_logging_setup - LogQueueHandler was not created or logger_instance_ref not set, skipping cleanup.") # Won't go to parent
            except Exception as e_handler_cleanup:
                # Use print to current sys.stderr for critical errors during cleanup itself if logging is dismantled
                current_err_stream = getattr(sys, 'stderr', sys.__stderr__)
                print(f"PipelineProcess: _undo_logging_setup - Error cleaning up LogQueueHandler: {e_handler_cleanup}", file=current_err_stream)

            # Restore original stdout/stderr only if they were changed
            # (Currently, they are not changed due to the diagnostic disabling above)
            if sys.stdout is not original_stdout:
                sys.stdout = original_stdout
                print("PipelineProcess: _undo_logging_setup - sys.stdout restored.", file=original_stdout)
            else:
                # Log to the current sys.stdout, which should be the original if not redirected
                print("PipelineProcess: _undo_logging_setup - sys.stdout was not redirected, no restoration needed.", file=sys.stdout)

            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
                print("PipelineProcess: _undo_logging_setup - sys.stderr restored.", file=original_stderr)
            else:
                print("PipelineProcess: _undo_logging_setup - sys.stderr was not redirected, no restoration needed.", file=sys.stderr)

            # print("PipelineProcess: _undo_logging_setup - Logging-specific cleanup finished.", file=original_stdout)
            # The above line might try to print to original_stdout which might be closed or None in some contexts
            # if it was never redirected. Better to use the current sys.stdout.
            print("PipelineProcess: _undo_logging_setup - Logging-specific cleanup finished.", file=sys.stdout)

        return _undo_logging_setup

    def _reset_logging_fields(self, request_id: str, stream_id: str):
        config_logging(request_id=request_id, stream_id=stream_id)
        # DIAGNOSTIC: current_queue_handler will be None, so this will effectively be skipped. (REVERTED - this comment may no longer be accurate)
        if hasattr(self, 'current_queue_handler') and self.current_queue_handler:
             config_logging_fields(self.current_queue_handler, request_id, stream_id)
        else:
            # This warning will not be sent to parent if log_queue is disabled. (REVERTED - this comment may no longer be accurate)
            logging.warning("PipelineProcess: _reset_logging_fields called but self.current_queue_handler is not set.")

    def _cleanup_process_queues(self):
        """Cleans up all multiprocessing queues from the child process side."""
        logging.info("PipelineProcess: _cleanup_process_queues - Starting general queue cleanup.")
        queues_to_clean = [
            ("log_queue", getattr(self, 'log_queue', None)),
            ("input_queue", getattr(self, 'input_queue', None)),
            ("output_queue", getattr(self, 'output_queue', None)),
            ("param_update_queue", getattr(self, 'param_update_queue', None)),
            ("error_queue", getattr(self, 'error_queue', None))
        ]

        for q_name, q_instance in queues_to_clean:
            if q_instance:
                try:
                    # For multiprocessing.Queue, close() should be called before join_thread().
                    # cancel_join_thread() is a more forceful way to ask the feeder thread to stop.
                    q_instance.close()
                    q_instance.cancel_join_thread()
                    logging.info(f"PipelineProcess: _cleanup_process_queues - {q_name}.close() and .cancel_join_thread() called. Attempting to join thread...")
                    # Attempt to join the queue's feeder thread.
                    # The feeder thread should exit after close() and cancel_join_thread() are called.
                    q_instance.join_thread() # Removed timeout argument
                    logging.info(f"PipelineProcess: _cleanup_process_queues - {q_name}.join_thread() completed.")
                except Exception as e_queue_cleanup:
                    # This might include queue.Empty or other errors if the queue is already in a bad state
                    logging.error(f"PipelineProcess: _cleanup_process_queues - Error cleaning up or joining {q_name}: {e_queue_cleanup}")
            else:
                logging.warning(f"PipelineProcess: _cleanup_process_queues - {q_name} not found on self for cleanup.")
        logging.info("PipelineProcess: _cleanup_process_queues - General queue cleanup finished.")

    def _queue_put_fifo(self, _queue: Union[mp.Queue, None], item: Any):
        """Helper to put an item on a queue, dropping oldest items if needed"""
        if not _queue: # Check if queue is None (can stay for robustness)
            # print(f"PipelineProcess: _queue_put_fifo - Attempted to put item on a None queue. Item: {item}", file=sys.stderr)
            return

        while not self.is_done():
            try:
                _queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    _queue.get_nowait()  # remove oldest item
                except queue.Empty:
                    continue

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
            self.process._queue_put_fifo(self.process.log_queue, text)

    def flush(self):
        self.original_stream.flush()

class LogQueueHandler(logging.Handler):
    """Send all log records to the process's log queue"""
    def __init__(self, process: PipelineProcess):
        super().__init__()
        self.process = process

    def emit(self, record):
        # DIAGNOSTIC: If log_queue is None, this handler should not be used by logging system,
        # but as a safeguard, check here too. (This check can remain for robustness)
        if not self.process.log_queue:
            # print(f"LogQueueHandler: emit - DIAGNOSTIC: process.log_queue is None. Record: {record.getMessage()}", file=sys.stderr)
            return
        msg = self.format(record)
        self.process._queue_put_fifo(self.process.log_queue, msg)

# Function to clear the queue
def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()  # Remove items without blocking
        except Exception as e:
            logging.error(f"Error while clearing queue: {e}")

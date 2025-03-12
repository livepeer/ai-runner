import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
import time
from typing import IO
from pydantic import BaseModel
import http.client

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError

class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()
        self.infer_script_path = (
            Path(__file__).parent.parent / "live" / "infer.py"
        )
        try:
            logging.info("Starting pipeline process")
            self.start_process(
                pipeline=self.model_id,  # we use the model_id as the pipeline name for now
                http_port=8888,
                # TODO: set torch device from self.torch_device
            )
        except Exception as e:
            raise InferenceError(original_exception=e)


    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, params: dict, request_id: str, stream_id: str, **kwargs
    ):
        if not self.process:
            raise RuntimeError("Pipeline process not running")

        max_retries = 10
        thrown_ex = None
        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(1)
            try:
                conn = http.client.HTTPConnection("localhost", 8888)
                conn.request(
                    "POST",
                    "/api/live-video-to-video",
                    json.dumps(
                        {
                            "subscribe_url": subscribe_url,
                            "publish_url": publish_url,
                            "control_url": control_url,
                            "events_url": events_url,
                            "params": params,
                            "request_id": request_id or "",
                            "stream_id": stream_id or "",
                        }
                    ),
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                if response.status != 200:
                    continue

                logging.info("Stream started successfully")
                return {} # Return an empty success response
            except Exception as e:
                thrown_ex = e
                logging.error(f"Attempt {attempt + 1} failed", exc_info=True)

        raise InferenceError(original_exception=thrown_ex)

    def get_health(self) -> HealthCheck:
        if not self.process:
            # The infer process is supposed to be always running, so if it's
            # gone it means an ERROR and the worker is allowed to kill us.
            return HealthCheck(status="ERROR")

        try:
            conn = http.client.HTTPConnection("localhost", 8888)
            conn.request("GET", "/api/status")
            response = conn.getresponse()

            if response.status != 200:
                raise ConnectionError(response.reason)

            # Re-declare just the field we need from PipelineStatus to avoid importing from ..live code
            class PipelineStatus(BaseModel):
                state: str = "OFFLINE"

            pipe_status = PipelineStatus(**json.loads(response.read().decode()))
            return HealthCheck(
                status="IDLE" if pipe_status.state == "OFFLINE" else "OK"
            )
        except Exception as e:
            logging.error(f"Failed to get status", exc_info=True)
            raise ConnectionError(f"Failed to get status: {e}")

    def start_process(self, **kwargs):
        cmd = ["python", str(self.infer_script_path)]

        # Add any additional kwargs as command-line arguments
        for key, value in kwargs.items():
            kebab_key = key.replace("_", "-")
            if isinstance(value, str):
                escaped_value = str(value).replace("'", "'\\''")
                cmd.extend([f"--{kebab_key}", f"{escaped_value}"])
            else:
                cmd.extend([f"--{kebab_key}", f"{value}"])

        env = os.environ.copy()
        env["HUGGINGFACE_HUB_CACHE"] = str(self.model_dir)

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )

            self.monitor_thread = threading.Thread(target=self.monitor_process)
            self.monitor_thread.start()
            self.log_thread = threading.Thread(target=log_output, args=(self.process.stdout,))
            self.log_thread.start()

        except subprocess.CalledProcessError as e:
            raise InferenceError(f"Error starting infer.py: {e}")

    def monitor_process(self):
        while True:
            if not self.process:
                logging.error("No process to monitor")
                return

            return_code: int
            try:
                return_code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.info("infer.py process is running...")
                continue
            except Exception:
                logging.error(f"Error while waiting for infer.py process to exit", exc_info=True)
                time.sleep(5)
                continue

            logging.info(f"infer.py process exited, cleaning up state... Return code: {return_code}")
            if return_code != 0:
                _, stderr = self.process.communicate()
                logging.error(
                    f"infer.py process failed with return code {return_code}. Error: {stderr}"
                )

            self.stop_process(is_monitor_thread=True)
            return

    def stop_process(self, is_monitor_thread: bool = False):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    logging.warning("Process did not terminate in time, force killing...")
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception as e:
                    logging.error(f"Error while force killing process: {e}")
                    os._exit(1)
            self.process = None
        if self.monitor_thread and not is_monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        if self.log_thread:
            self.log_thread.join()
            self.log_thread = None
        logging.info("Infer process stopped successfully")

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"


def log_output(f: IO[str]):
    try:
        for line in f:
            sys.stderr.write(line)
    except Exception as e:
        logging.error(f"Error while logging process output: {e}")

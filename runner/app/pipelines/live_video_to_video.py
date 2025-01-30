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

logger = logging.getLogger(__name__)


class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()
        self.infer_script_path = (
            Path(__file__).parent.parent / "live" / "infer.py"
        )
        self.process = None
        self.monitor_thread = None
        self.log_thread = None


    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, stream_id: str, params: dict, **kwargs
    ):
        if self.process:
            raise RuntimeError("Pipeline already running")

        try:
            logger.info(f"Starting stream, subscribe={subscribe_url} publish={publish_url}, control={control_url}, events={events_url}, stream_id={stream_id}")
            self.start_process(
                pipeline=self.model_id,  # we use the model_id as the pipeline name for now
                http_port=8888,
                subscribe_url=subscribe_url,
                publish_url=publish_url,
                control_url=control_url,
                events_url=events_url,
                stream_id=stream_id,
                initial_params=json.dumps(params),
                # TODO: set torch device from self.torch_device
            )
            return
        except Exception as e:
            raise InferenceError(original_exception=e)

    def get_health(self) -> HealthCheck:
        if not self.process:
            return HealthCheck(status="IDLE")

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
            health_status = "OK"
            if pipe_status.state == "OFFLINE":
                # The infer process is supposed to exit when it goes offline, so if we get this status it means an error
                # and the worker is allowed to kill us.
                health_status = "ERROR"

            return HealthCheck(status=health_status)
        except Exception as e:
            logger.error(f"Failed to get status", exc_info=True)
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
                logger.error("No process to monitor")
                return

            return_code: int
            try:
                return_code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.info("infer.py process is running...")
                continue
            except Exception:
                logger.error(f"Error while waiting for infer.py process to exit", exc_info=True)
                time.sleep(5)
                continue

            logger.info(f"infer.py process exited, cleaning up state... Return code: {return_code}")
            if return_code != 0:
                _, stderr = self.process.communicate()
                logger.error(
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
                    logger.warning("Process did not terminate in time, force killing...")
                    self.process.kill()
                    self.process.wait(timeout=5)
                except Exception as e:
                    logger.error(f"Error while force killing process: {e}")
                    os._exit(1)
            self.process = None
        if self.monitor_thread and not is_monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        if self.log_thread:
            self.log_thread.join()
            self.log_thread = None
        logger.info("Infer process stopped successfully")

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"


def log_output(f: IO[str]):
    try:
        for line in f:
            sys.stderr.write(f"[infer.py] {line}")
    except Exception as e:
        logger.error(f"Error while logging process output: {e}")

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
import psutil

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError

proc_status_important_fields = ["State", "VmRSS", "VmSize", "Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "CoreDumping"]

class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.version = os.getenv("VERSION", "undefined")
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()
        self.infer_script_path = (
            Path(__file__).parent.parent / "live" / "infer.py"
        )
        self.restart_count = 0
        self.start_process()

    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, params: dict, request_id: str, manifest_id: str, stream_id: str, **kwargs
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
                            "manifest_id": manifest_id or "",
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
            logging.error("[HEALTHCHECK] Infer process is not running")
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
                status=(
                    "IDLE" if pipe_status.state == "OFFLINE"
                    else "ERROR" if pipe_status.state == "ERROR"
                    else "OK"
                ),
            )
        except Exception as e:
            logging.error(f"[HEALTHCHECK] Failed to get status: {type(e).__name__}: {str(e)}")
            # Diagnostics might take a while to collect CPU metrics, so we do it in a background thread
            threading.Thread(target=lambda: self.log_process_diagnostics(full=True)).start()
            raise ConnectionError(f"Failed to get status: {e}")

    def start_process(self):
        logging.info("Starting pipeline process")
        cmd = [sys.executable, str(self.infer_script_path)]
        cmd.extend(["--pipeline", self.model_id]) # we use the model_id as the pipeline name for now
        cmd.extend(["--http-port", "8888"])
        # TODO: set torch device from self.torch_device

        env = os.environ.copy()
        env["HUGGINGFACE_HUB_CACHE"] = str(self.model_dir)

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )

            self.monitor_thread = threading.Thread(target=self.monitor_process)
            self.monitor_thread.start()
            self.log_thread = threading.Thread(target=log_output, daemon=True, args=(self.process.stdout,))
            self.log_thread.start()

        except subprocess.CalledProcessError as e:
            raise InferenceError(f"Error starting infer.py: {e}")

    def monitor_process(self):
        # Wait 1 sec before starting to monitor the process. This gives it some
        # time to start and also ensures we won't restart the process too often.
        time.sleep(1)

        while True:
            if not self.process:
                logging.error("No process to monitor")
                return

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                self.log_process_diagnostics(logging.DEBUG)

            try:
                return_code = self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logging.info("infer.py process is running...")
                continue
            except Exception:
                logging.error(f"Error while waiting for infer.py process to exit", exc_info=True)
                time.sleep(5)
                continue

            logging.info(f"infer.py process exited with return_code={return_code}")
            self.log_process_diagnostics(full=True)
            break

        self.restart_count += 1
        if self.restart_count > 10:
            logging.error("infer.py process has restarted more than 10 times. Exiting.")
            os._exit(1)

        # Start a separate thread to restart the process since it will
        # restart the monitor thread itself (the current thread).
        def restart_process():
            try:
                logging.info(f"Restarting infer.py process restart_count={self.restart_count}")
                self.stop_process()
                self.start_process()
            except Exception as e:
                logging.error(f"Error restarting infer.py process: {e}")
                os._exit(1)
        threading.Thread(target=restart_process).start()

    def stop_process(self):
        if self.process:
            if self.process.stdout:
                # Closing the output stream sometimes hangs, so we do it in a separate daemon thread
                # and join the log_thread below with a timeout. If it does hang there might be a thread
                # leak which is why we limit to up to 10 restarts.
                stdout = self.process.stdout
                threading.Thread(target=lambda: stdout.close(), daemon=True).start()
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
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        if self.log_thread:
            self.log_thread.join(timeout=1)
            if self.log_thread.is_alive():
                logging.warning("Log thread did not finish")
            self.log_thread = None
        logging.info("Infer process stopped successfully")


    def log_process_diagnostics(self, level: int = logging.INFO, full: bool = False):
        """Collect and log process diagnostics. To be called from a background thread if needed.
        """
        try:
            diagnostics = self.collect_process_diagnostics(full=full)
            logging.log(level, f"infer.py process diagnostics={json.dumps(diagnostics)}")
        except:
            logging.exception(f"Error collecting infer.py process diagnostics")

    def collect_process_diagnostics(self, full: bool = False):
        """Collect process diagnostics using different tools and returns a dict with the results

        Args:
            full: If True, collect full diagnostic information, otherwise just key fields
        """
        # Get system info
        system_info = {}
        try:
            system_info = {
                "memory": psutil.virtual_memory()._asdict(),
                "cpu": psutil.cpu_percent(interval=0.1, percpu=True),
                "disk": psutil.disk_usage('/')._asdict()
            }
        except:
            logging.exception("Failed to collect system diagnostics")

        if not self.process:
            return {"system_info": system_info, "process_info": {"is_running": False}}

        # Get process info
        pid = self.process.pid
        process_info = {
            "pid": pid,
            "return_code": self.process.poll(),
        }

        try:
            if not psutil.pid_exists(pid):
                logging.error("Process ID doesn't exist in psutil")
            else:
                p = psutil.Process(pid)
                process_info = {
                    **process_info,
                    "memory_info": p.memory_info()._asdict(),
                    "cpu_percent": p.cpu_percent(),
                    "create_time": p.create_time(),
                    "status": p.status(),
                    "is_running": p.is_running()
                }
        except:
            logging.exception("Failed to collect psutil diagnostics")

        # Collect /proc information
        def read_proc_as_map(path: str) -> dict | str:
            try:
                map = {}
                with open(path, "r") as f:
                    for line in f:
                        key, value = line.strip().split(':', 1)
                        map[key] = value.strip()
                return map
            except:
                # return the file as a string if it's not parseable as a map
                with open(path, "r") as f:
                    return f.read()

        os_proc_info: dict[str, str | dict] = {}
        for proc_file in ["status", "wchan", "io"]:
            try:
                path = f"/proc/{pid}/{proc_file}"
                if not os.path.exists(path):
                    os_proc_info[proc_file] = "File does not exist"
                    continue

                info = read_proc_as_map(path)
                if proc_file == "status" and not full and isinstance(info, dict):
                    info = {k: v for k, v in info.items() if k in proc_status_important_fields}
                os_proc_info[proc_file] = info
            except Exception as e:
                logging.exception(f"Failed to read /proc/{pid}/{proc_file}")

        return {
            "system_info": system_info,
            "process_info": process_info,
            "os_proc_info": os_proc_info,
        }

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"

def log_output(f: IO[str]):
    try:
        for line in f:
            sys.stderr.write(line)
    except Exception as e:
        logging.error(f"Error while logging process output: {e}")

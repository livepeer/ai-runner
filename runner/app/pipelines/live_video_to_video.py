import json
import logging
import os
import threading
import time
from pydantic import BaseModel
import psutil

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError
from app.live.live_infer_app import LiveInferApp, StreamParams

proc_status_important_fields = ["State", "VmRSS", "VmSize", "Threads", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "CoreDumping"]

class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.version = os.getenv("VERSION", "undefined")
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()

        initial_params_env = os.environ.get("INFERPY_INITIAL_PARAMS")
        try:
            initial_params = json.loads(initial_params_env) if initial_params_env else {}
        except Exception as e:
            logging.error(f"Error parsing INFERPY_INITIAL_PARAMS: {e}")
            initial_params = {}

        self.app = LiveInferApp(pipeline=self.model_id, initial_params=initial_params)
        # Run async app.start() in a dedicated loop thread for sync callers
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever, name="LiveInferAppLoop", daemon=False)
        self.loop_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self.app.start(), self.loop)
        fut.result(timeout=30)

    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, params: dict, request_id: str, manifest_id: str, stream_id: str, **kwargs
    ):
        try:
            sp = StreamParams(
                subscribe_url=subscribe_url,
                publish_url=publish_url,
                control_url=control_url,
                events_url=events_url,
                params=params,
                request_id=request_id or "",
                manifest_id=manifest_id or "",
                stream_id=stream_id or "",
            )
            fut = asyncio.run_coroutine_threadsafe(self.app.start_stream(sp), self.loop)
            fut.result(timeout=30)
            logging.info("Stream started successfully")
            return {}
        except Exception as e:
            logging.error("Failed to start stream", exc_info=True)
            raise InferenceError(original_exception=e)

    def get_health(self) -> HealthCheck:
        try:
            fut = asyncio.run_coroutine_threadsafe(self.app.get_status(), self.loop)
            status = fut.result(timeout=5)

            # Re-declare just the field we need without importing live models
            class PipelineStatus(BaseModel):
                state: str = "OFFLINE"

            pipe_status = PipelineStatus(**status.model_dump())
            return HealthCheck(
                status=(
                    "LOADING" if pipe_status.state == "LOADING"
                    else "IDLE" if pipe_status.state == "OFFLINE"
                    else "ERROR" if pipe_status.state == "ERROR"
                    else "OK"
                ),
            )
        except Exception as e:
            logging.error(f"[HEALTHCHECK] Failed to get status: {type(e).__name__}: {str(e)}")
            threading.Thread(target=lambda: self.log_process_diagnostics(full=True)).start()
            raise ConnectionError(f"Failed to get status: {e}")

    # Removed subprocess-based lifecycle; LiveInferApp owns runtime in-process


    def log_process_diagnostics(self, level: int = logging.INFO, full: bool = False):
        """Collect and log diagnostics for the current process. To be called from a background thread if needed."""
        try:
            diagnostics = self.collect_process_diagnostics(full=full)
            logging.log(level, f"live infer diagnostics={json.dumps(diagnostics)}")
        except:
            logging.exception(f"Error collecting live infer diagnostics")

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

        # Get process info (current process)
        pid = os.getpid()
        process_info = {
            "pid": pid,
            "return_code": None,
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

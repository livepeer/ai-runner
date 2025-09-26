import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Optional
import signal
import threading

from pydantic import BaseModel, Field, model_validator

from .log import config_logging, log_timing
from .process import ProcessGuardian
from .streamer import PipelineStreamer
from .streamer.protocol import TrickleProtocol
from .trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT


class StreamParams(BaseModel):
    subscribe_url: str
    publish_url: str
    control_url: str = ""
    events_url: str = ""
    params: dict = Field(default_factory=dict)
    request_id: str = ""
    manifest_id: str = ""
    stream_id: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    @model_validator(mode="after")
    def ensure_dimensions(self):
        p = dict(self.params or {})
        p.setdefault("width", self.width)
        p.setdefault("height", self.height)
        self.params = p
        return self


class LiveInferApp:
    """
    Async runtime for live inference. Manages `ProcessGuardian` and `PipelineStreamer` lifecycles.
    Callers should await methods from their own event loop.
    """

    def __init__(self, *, pipeline: str, initial_params: Optional[dict] = None):
        self.pipeline = pipeline
        self.initial_params = initial_params or {}
        self._process: Optional[ProcessGuardian] = None
        self._streamer: Optional[PipelineStreamer] = None
        self._last_params_file = os.path.join(
            tempfile.gettempdir(), "ai_runner_last_params.json"
        )

    async def start(self):
        await self._cleanup_last_stream()
        self._process = ProcessGuardian(self.pipeline, self.initial_params)
        with log_timing("starting ProcessGuardian"):
            await self._process.start()

    async def stop(self):
        stop_coros = []
        if self._streamer:
            try:
                self._streamer.trigger_stop_stream()
                stop_coros.append(self._streamer.wait())
            except Exception:
                logging.exception("Error triggering streamer stop")
        if self._process:
            stop_coros.append(self._process.stop())

        if not stop_coros:
            return
        results = await asyncio.wait_for(asyncio.gather(*stop_coros, return_exceptions=True), timeout=10)
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            raise ExceptionGroup("Error stopping components", exceptions)
        self._streamer = None
        self._process = None

    async def start_stream(self, sp: StreamParams):
        if not self._process:
            raise RuntimeError("Process not running")

        stream_request_timestamp = int(time.time() * 1000)

        if self._streamer and self._streamer.is_running():
            try:
                self._streamer.trigger_stop_stream()
                await self._streamer.wait(timeout=3)
            except asyncio.TimeoutError:
                raise RuntimeError("Timeout stopping previous streamer")

        try:
            with open(self._last_params_file, "w") as f:
                json.dump(sp.model_dump(), f)
        except Exception:
            logging.exception("Error saving last params to file")

        config_logging(request_id=sp.request_id, manifest_id=sp.manifest_id, stream_id=sp.stream_id)

        width = sp.params.get("width", DEFAULT_WIDTH)
        height = sp.params.get("height", DEFAULT_HEIGHT)
        if self.pipeline == "comfyui":
            width = height = 512
            sp.params = sp.params | {"width": width, "height": height}
            logging.warning("Using default dimensions for ComfyUI pipeline")

        protocol = TrickleProtocol(
            sp.subscribe_url, sp.publish_url, sp.control_url, sp.events_url, width, height
        )
        self._streamer = PipelineStreamer(
            protocol, self._process, sp.request_id, sp.manifest_id, sp.stream_id
        )

        await self._streamer.start(sp.params)
        await protocol.emit_monitoring_event(
            {"type": "runner_receive_stream_request", "timestamp": stream_request_timestamp},
            queue_event_type="stream_trace",
        )

    async def update_params(self, params: dict):
        if not self._process:
            raise RuntimeError("Process not running")
        await self._process.update_params(params)

    async def get_status(self):
        if not self._process:
            raise RuntimeError("Process not running")
        return self._process.get_status()

    async def _cleanup_last_stream(self):
        if not os.path.exists(self._last_params_file):
            return
        try:
            with open(self._last_params_file, "r") as f:
                params = json.load(f)
            os.remove(self._last_params_file)

            protocol = TrickleProtocol(
                params.get("subscribe_url", ""),
                params.get("publish_url", ""),
                params.get("control_url", ""),
                params.get("events_url", ""),
            )
            await protocol.start()
            await protocol.stop()
        except Exception:
            logging.exception("Error cleaning up last stream trickle channels")

    def setup_fatal_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Opt-in installation of simple fatal handlers:
        - Loop exception handler schedules app.stop()
        - threading.excepthook schedules app.stop()
        - SIGINT/SIGTERM schedule app.stop() (only from main thread)
        No additional state is tracked here.
        """
        target_loop = loop or asyncio.get_running_loop()

        def schedule_stop():
            try:
                target_loop.create_task(self.stop())
            except RuntimeError:
                # Loop may be closing; best-effort
                pass

        def loop_exception_handler(_loop, context):
            logging.error(
                "Unhandled exception in asyncio task",
                exc_info=context.get("exception"),
            )
            schedule_stop()

        target_loop.set_exception_handler(loop_exception_handler)

        original_hook = threading.excepthook

        def thread_hook(args):
            logging.error(
                "Unhandled exception in thread",
                exc_info=args.exc_value,
            )
            try:
                target_loop.call_soon_threadsafe(schedule_stop)
            except Exception:
                pass
            original_hook(args)

        threading.excepthook = thread_hook

        if threading.current_thread() is not threading.main_thread():
            logging.debug("Skipping signal handlers: not on main thread")
            return

        def _signal_handler(sig, _):
            logging.info(f"Received signal {sig}, scheduling graceful stop")
            try:
                target_loop.call_soon_threadsafe(schedule_stop)
            except Exception:
                pass

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            logging.exception("Failed to install signal handlers")


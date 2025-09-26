import asyncio
import json
import logging
import os
import signal
import tempfile
import threading
import time
from typing import Optional, List

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
	Inline runtime for live inference with a private asyncio loop and synchronous API.
	Manages `ProcessGuardian` and `PipelineStreamer` lifecycles and preserves shutdown ordering/timeouts.
	Handles exceptions and signals; catastrophic errors terminate the process. Cleans up orphaned Trickle channels on startup.
	"""

    def __init__(self, *, pipeline: str, initial_params: Optional[dict] = None):
        self.pipeline = pipeline
        self.initial_params = initial_params or {}

        # Event loop in background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # Core components
        self._process: Optional[ProcessGuardian] = None
        self._streamer: Optional[PipelineStreamer] = None
        # Lifecycle and error signaling
        self._uncaught_event: Optional[asyncio.Event] = None
        self._signal_event: Optional[asyncio.Event] = None
        self._fatal_shutdown: bool = False
        self._lifecycle_task: Optional[asyncio.Task] = None

        # Shutdown completion for synchronous wait
        self._stopped = threading.Event()
        self._loop_ready = threading.Event()

        # File used to cleanup leftover trickle channels from a previous crash
        self._last_params_file = os.path.join(
            tempfile.gettempdir(), "ai_runner_last_params.json"
        )

    # ------------------------------ Public sync API ------------------------------
    def start(self) -> None:
        """
        Start the private loop and ProcessGuardian, install handlers, cleanup orphaned channels.
        """
        if self._loop is not None:
            return

        self._stopped.clear()

        def _run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._install_loop_exception_handler()
            self._uncaught_event = asyncio.Event()
            self._signal_event = asyncio.Event()
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._thread = threading.Thread(target=_run_loop, name="LiveInferAppLoop", daemon=False)
        self._thread.start()

        # Ensure loop is ready before scheduling work onto it
        if not self._loop_ready.wait(timeout=5):
            raise RuntimeError("Failed to initialize LiveInferApp loop")

        # Install global handlers (must run on caller thread)
        self._install_thread_exception_hook()
        self._install_signal_handlers()

        # Configure base logging level from VERBOSE_LOGGING
        log_level = logging.DEBUG if os.getenv("VERBOSE_LOGGING") == "1" else logging.INFO
        config_logging(log_level=log_level)

        # Async init on the loop
        self._run_sync(self._async_start(), timeout=30)

    def start_stream(self, stream_params: StreamParams) -> None:
        self._run_sync(self._async_start_stream(stream_params), timeout=30)

    def update_params(self, params: dict) -> None:
        self._run_sync(self._async_update_params(params), timeout=10)

    def get_status(self):
        return self._run_sync(self._async_get_status(), timeout=5)

    def stop(self) -> None:
        if not self._loop:
            return
        self._run_sync(self._async_stop(), timeout=15)
        self._stop_loop()

    def wait(self, timeout: float | None = None) -> None:
        self._stopped.wait(timeout=timeout)

    # ------------------------------ Internal async impl ------------------------------
    async def _async_start(self):
        # Cleanup last stream trickle channels (if any)
        await self._cleanup_last_stream()

        # Start ProcessGuardian
        self._process = ProcessGuardian(self.pipeline, self.initial_params)
        with log_timing("starting ProcessGuardian"):
            await self._process.start()

        # Spawn lifecycle supervisor
        self._lifecycle_task = asyncio.create_task(self._lifecycle_supervisor())

    async def _async_stop(self):
        stop_coros: List[asyncio._CoroutineLike] = []
        if self._streamer:
            try:
                self._streamer.trigger_stop_stream()
                stop_coros.append(self._streamer.wait())
            except Exception:
                # Defensive: ensure stop continues
                logging.exception("Error triggering streamer stop")
        if self._process:
            stop_coros.append(self._process.stop())
        if self._http_runner:
            stop_coros.append(self._http_runner.cleanup())

        try:
            if not stop_coros:
                return
            gathered = asyncio.gather(*stop_coros, return_exceptions=True)
            results = await asyncio.wait_for(gathered, timeout=10)
            exceptions = [r for r in results if isinstance(r, Exception)]
            if exceptions:
                raise ExceptionGroup("Error stopping components", exceptions)
        except Exception:
            logging.error("Graceful shutdown error, exiting abruptly", exc_info=True)
            os._exit(1)
        finally:
            self._streamer = None
            self._process = None
            self._http_runner = None
            self._stopped.set()

    async def _async_start_stream(self, sp: StreamParams):
        if not self._process:
            raise RuntimeError("Process not running")

        stream_request_timestamp = int(time.time() * 1000)

        # Stop previous streamer if running
        if self._streamer and self._streamer.is_running():
            try:
                logging.info("Stopping previous streamer")
                self._streamer.trigger_stop_stream()
                await self._streamer.wait(timeout=3)
            except asyncio.TimeoutError:
                logging.error("Timeout stopping previous streamer")
                raise RuntimeError("Timeout stopping previous streamer")

        # Persist last params for post-crash cleanup parity
        try:
            with open(self._last_params_file, "w") as f:
                json.dump(sp.model_dump(), f)
        except Exception as e:
            logging.error(f"Error saving last params to file: {e}")

        # Configure request-scoped logging fields
        config_logging(request_id=sp.request_id, manifest_id=sp.manifest_id, stream_id=sp.stream_id)

        # Dimension defaults and ComfyUI override
        width = sp.params.get("width", DEFAULT_WIDTH)
        height = sp.params.get("height", DEFAULT_HEIGHT)
        if self.pipeline == "comfyui":
            width = height = 512
            sp.params = sp.params | {"width": width, "height": height}
            logging.warning("Using default dimensions for ComfyUI pipeline")
        else:
            logging.info(f"Using dimensions from params: {width}x{height}")

        # Protocol and streamer
        protocol = TrickleProtocol(
            sp.subscribe_url, sp.publish_url, sp.control_url, sp.events_url, width, height
        )
        self._streamer = PipelineStreamer(
            protocol, self._process, sp.request_id, sp.manifest_id, sp.stream_id
        )

        await self._streamer.start(sp.params)
        await protocol.emit_monitoring_event(
            {
                "type": "runner_receive_stream_request",
                "timestamp": stream_request_timestamp,
            },
            queue_event_type="stream_trace",
        )

    async def _async_update_params(self, params: dict):
        if not self._process:
            raise RuntimeError("Process not running")
        await self._process.update_params(params)

    async def _async_get_status(self):
        if not self._process:
            raise RuntimeError("Process not running")
        return self._process.get_status()

    async def _cleanup_last_stream(self):
        # Best-effort cleanup of orphaned trickle channels
        if not os.path.exists(self._last_params_file):
            logging.debug("No last stream params found to cleanup")
            return

        try:
            with open(self._last_params_file, "r") as f:
                params = json.load(f)
            os.remove(self._last_params_file)

            logging.info(
                f"Cleaning up last stream trickle channels for request_id={params.get('request_id','')} subscribe_url={params.get('subscribe_url','')} publish_url={params.get('publish_url','')} control_url={params.get('control_url','')} events_url={params.get('events_url','')}"
            )
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

    async def _lifecycle_supervisor(self):
        assert self._uncaught_event and self._signal_event
        try:
            await asyncio.wait(
                [
                    asyncio.create_task(self._signal_event.wait()),
                    asyncio.create_task(self._uncaught_event.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:
            pass
        finally:
            try:
                await self._async_stop()
            finally:
                os._exit(1)

    # ------------------------------ Handlers & utilities ------------------------------
    def _install_loop_exception_handler(self):
        def asyncio_exception_handler(loop, context):
            exception = context.get("exception")
            logging.error(
                f"Terminating process due to unhandled exception in asyncio task: {exception}",
                exc_info=exception,
            )
            try:
                if self._uncaught_event:
                    self._fatal_shutdown = True
                    self._uncaught_event.set()
            except Exception:
                os._exit(1)

        assert self._loop is not None
        self._loop.set_exception_handler(asyncio_exception_handler)

    def _install_thread_exception_hook(self):
        original_hook = threading.excepthook

        def custom_hook(args):
            logging.error(
                f"Terminating process due to unhandled exception in thread: {args.exc_value}",
                exc_info=args.exc_value,
            )
            try:
                if self._loop and self._uncaught_event:
                    self._fatal_shutdown = True
                    self._loop.call_soon_threadsafe(self._uncaught_event.set)
            except Exception:
                os._exit(1)
            original_hook(args)

        threading.excepthook = custom_hook

    def _install_signal_handlers(self):
        # Must be installed from main thread
        if threading.current_thread() is not threading.main_thread():
            logging.debug("Signal handlers not installed: not on main thread")
            return
        assert self._loop is not None and self._signal_event is not None

        def _signal_handler(sig, _):
            if self._loop and self._signal_event:
                try:
                    self._loop.call_soon_threadsafe(self._signal_event.set)
                except Exception:
                    os._exit(1)

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            logging.exception("Failed to install signal handlers from main thread")

    def _stop_loop(self):
        if not self._loop:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        finally:
            if self._thread:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None

    def _run_sync(self, coro, timeout: float):
        if not self._loop:
            raise RuntimeError("LiveInferApp not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------ Async API for HTTP facade ------------------------------
    # Expose async methods to be used by aiohttp handlers (executed on our loop)
    async def a_start_stream(self, **kwargs):
        return await self._async_start_stream(**kwargs)

    async def a_update_params(self, params: dict):
        return await self._async_update_params(params)

    async def a_get_status(self):
        return await self._async_get_status()


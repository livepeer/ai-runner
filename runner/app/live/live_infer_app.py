import asyncio
import json
import logging
import os
import time
from typing import Optional, TypeVar, Callable, Coroutine, Any
import signal
import threading

from pydantic import BaseModel, Field
from typing import Annotated, Dict

from .log import config_logging, log_timing
from .process import ProcessGuardian
from .pipelines.interface import BaseParams
from .streamer import PipelineStreamer
from .streamer.protocol import TrickleProtocol


class StreamParams(BaseModel):
    subscribe_url: Annotated[
        str,
        Field(
            ...,
            description="Source URL of the incoming stream to subscribe to.",
        ),
    ]
    publish_url: Annotated[
        str,
        Field(
            ...,
            description="Destination URL of the outgoing stream to publish.",
        ),
    ]
    control_url: Annotated[
        str,
        Field(
            default="",
            description="URL for subscribing via Trickle protocol for updates in the live video-to-video generation params.",
        ),
    ]
    events_url: Annotated[
        str,
        Field(
            default="",
            description="URL for publishing events via Trickle protocol for pipeline status and logs.",
        ),
    ]
    params: Annotated[
        Dict,
        Field(default={}, description="Initial parameters for the pipeline."),
    ]
    request_id: Annotated[
        str,
        Field(default="", description="Unique identifier for the request."),
    ]
    manifest_id: Annotated[
        str,
        Field(default="", description="Orchestrator identifier for the request."),
    ]
    stream_id: Annotated[
        str,
        Field(default="", description="Unique identifier for the stream."),
    ]


class LiveInferApp:
    """
    Async runtime for live inference. Manages `ProcessGuardian` and `PipelineStreamer` lifecycles.
    Callers should await methods from their own event loop.
    """

    def __init__(
        self, *, pipeline: str, initial_params: dict, model_dir: Optional[str] = None
    ):
        if model_dir is None:
            model_dir = os.environ.get("MODEL_DIR", "./models")

        self.pipeline = pipeline
        self.initial_params = initial_params
        self.model_dir = model_dir

        self._process = ProcessGuardian(self.pipeline, self.initial_params, self.model_dir)
        self._streamer: Optional[PipelineStreamer] = None
        self._last_params_file = os.path.join(self.model_dir, "ai-runner-aux-data", "last-params.json")
        self._stopped = asyncio.Event()

    async def start(self):
        config_logging(log_level=logging.DEBUG if os.getenv("VERBOSE_LOGGING")=="1" else logging.INFO)
        self._stopped.clear()
        await self._cleanup_last_stream()
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

        try:
            if not stop_coros:
                return
            results = await asyncio.wait_for(asyncio.gather(*stop_coros, return_exceptions=True), timeout=10)
            exceptions = [r for r in results if isinstance(r, Exception)]
            if exceptions:
                raise ExceptionGroup("Error stopping components", exceptions)
        finally:
            self._streamer = None
            self._stopped.set()

    async def wait_for_stop(self):
        await self._stopped.wait()

    async def start_stream(self, params: StreamParams):
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
            os.makedirs(os.path.dirname(self._last_params_file), exist_ok=True)
            with open(self._last_params_file, "w") as f:
                json.dump(params.model_dump(), f)
        except Exception:
            logging.exception(f"Error saving last params to file={self._last_params_file} params={params}")

        config_logging(request_id=params.request_id, manifest_id=params.manifest_id, stream_id=params.stream_id)

        if self.pipeline == "comfyui":
            logging.warning("Using default dimensions for ComfyUI pipeline")
            params = params.model_copy()
            params.params = params.params | {"width": 512, "height": 512}

        proto_params = BaseParams(**params.params)
        protocol = TrickleProtocol(
            params.subscribe_url, params.publish_url, params.control_url, params.events_url, proto_params.width, proto_params.height
        )
        self._streamer = PipelineStreamer(
            protocol, self._process, params.request_id, params.manifest_id, params.stream_id
        )

        await self._streamer.start(params.params)
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

    def setup_fatal_signal_handlers(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Opt-in installation of simple fatal handlers (uncaught exceptions and signals).
        When any such events are detected, we schedule a graceful stop of the app.
        """
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Fatal signal handlers must be installed on main thread")

        target_loop = loop or asyncio.get_running_loop()

        async def do_stop():
            try:
                await self.stop()
            except Exception:
                logging.exception("Error stopping app, crashing process", exc_info=True)
                os._exit(1)

        def trigger_stop():
            try:
                asyncio.run_coroutine_threadsafe(do_stop(), target_loop)
            except Exception:
                # loop might be already shutting down, ignore
                pass

        def _loop_exception_handler(_loop, context):
            logging.error("Terminating process due to unhandled exception in asyncio task", exc_info=context.get("exception"))
            trigger_stop()

        target_loop.set_exception_handler(_loop_exception_handler)

        original_hook = threading.excepthook
        def _thread_hook(args):
            logging.error("Terminating process due to unhandled exception in thread", exc_info=args.exc_value)
            trigger_stop()
            original_hook(args)

        threading.excepthook = _thread_hook

        def _signal_handler(sig, _):
            logging.info(f"Received signal {sig}, scheduling graceful stop")
            trigger_stop()

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            logging.exception("Failed to install signal handlers")

    async def _cleanup_last_stream(self):
        """
        Cleans up the last stream trickle channels. Uses a temporary file to store the last stream params every time a
        stream is started. So even if the app restarts, it can be found and gracefully stopped.
        """
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
            # Simply start and stop the protocol to clean up the trickle channels
            await protocol.start()
            await protocol.stop()
        except Exception:
            logging.exception("Error cleaning up last stream trickle channels")

T = TypeVar("T")

class SyncLiveInferApp:
    """
    Sync wrapper for LiveInferApp. This class runs the LiveInferApp in a dedicated asyncio loop thread for using it
    outside of an asyncio loop. It allows calling async methods from the main thread.
    """

    def __init__(self, *, pipeline: str, initial_params: dict, model_dir: Optional[str] = None):
        self.__app = LiveInferApp(pipeline=pipeline, initial_params=initial_params, model_dir=model_dir)

        self.loop = asyncio.new_event_loop()
        def run_loop():
            try:
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            finally:
                self.loop.close()
        self.loop_thread = threading.Thread(target=run_loop, name="LiveInferAppLoop", daemon=False)
        self.stopped = True

    def setup_fatal_signal_handlers(self):
        self.__app.setup_fatal_signal_handlers(self.loop)

    def start(self):
        if not self.stopped:
            raise RuntimeError("Already started")
        self.stopped = False

        self.loop_thread.start()
        self._run(lambda app: app.start())

    def start_stream(self, stream_params: StreamParams):
        return self._run(lambda app: app.start_stream(stream_params))

    def get_status(self):
        return self._run(lambda app: app.get_status())

    def stop(self):
        if self.stopped:
            raise RuntimeError("Already stopped")
        self.stopped = True

        self._run(lambda app: app.stop())
        self.loop.stop()
        self.loop_thread.join()

    def _run(self, fn: Callable[[LiveInferApp], Coroutine[Any, Any, T]]) -> T:
        future = asyncio.run_coroutine_threadsafe(fn(self.__app), self.loop)
        return future.result()

"""
Old-school CLI and thin HTTP facade for running the inline LiveInferApp.

This module exposes:
- InferAPI: aiohttp handlers that delegate to LiveInferApp methods
- CLI entrypoint: starts LiveInferApp and serves the HTTP API

Note: The main pipeline integrates LiveInferApp inline and does not use this CLI.
This exists for parity/testing and should remain a thin delegation layer only.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Annotated, Dict, Optional, cast

from aiohttp import web
from pydantic import BaseModel, Field

from .live_infer_app import LiveInferApp
from .log import config_logging


class StartStreamParams(BaseModel):
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


class InferAPI:
    def __init__(self, live_infer_app: LiveInferApp):
        self.live_infer_app = live_infer_app
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Routes
        self.app.router.add_post("/api/live-video-to-video", self.handle_start_stream)
        self.app.router.add_post("/api/params", self.handle_params_update)
        self.app.router.add_get("/api/status", self.handle_get_status)

    async def _parse_request_data(self, request: web.Request) -> Dict:
        if request.content_type.startswith("application/json"):
            return await request.json()
        else:
            raise ValueError(f"Unknown content type: {request.content_type}")

    async def handle_start_stream(self, request: web.Request):
        try:
            params_data = await self._parse_request_data(request)
            params = StartStreamParams(**params_data)

            # Configure request-scoped logging fields
            config_logging(
                request_id=params.request_id,
                manifest_id=params.manifest_id,
                stream_id=params.stream_id,
            )

            # Delegate to LiveInferApp
            await asyncio.to_thread(self.live_infer_app.start_stream, StreamParams(
                subscribe_url=params.subscribe_url,
                publish_url=params.publish_url,
                control_url=params.control_url,
                events_url=params.events_url,
                params=params.params,
                request_id=params.request_id,
                manifest_id=params.manifest_id,
                stream_id=params.stream_id,
            ))
            return web.Response(text="Stream started successfully")
        except Exception as e:
            logging.error(f"Error starting stream: {e}")
            return web.Response(text=f"Error starting stream: {str(e)}", status=400)

    async def handle_params_update(self, request: web.Request):
        try:
            params = await self._parse_request_data(request)

            await asyncio.to_thread(self.live_infer_app.update_params, params)
            return web.Response(text="Params updated successfully")
        except Exception as e:
            logging.error(f"Error updating params: {e}")
            return web.Response(text=f"Error updating params: {str(e)}", status=400)

    async def handle_get_status(self, request: web.Request):
        status = await asyncio.to_thread(self.live_infer_app.get_status)
        return web.json_response(status.model_dump())

    @classmethod
    async def serve(cls, port: int, live_infer_app: LiveInferApp):
        api = cls(live_infer_app)
        api.runner = web.AppRunner(api.app)
        await api.runner.setup()
        api.site = web.TCPSite(api.runner, "0.0.0.0", port)
        await api.site.start()
        logging.info(f"HTTP server started on port {port}")
        return api

    async def shutdown(self):
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            self.site = None


async def main():
    port = int(os.environ.get("INFER_HTTP_PORT", "8888"))
    pipeline = os.environ.get("PIPELINE", "comfyui")
    initial_params_env = os.environ.get("INFERPY_INITIAL_PARAMS", "{}")
    try:
        initial_params = json.loads(initial_params_env) if initial_params_env else {}
    except Exception as e:
        logging.error(f"Error parsing INFERPY_INITIAL_PARAMS: {e}")
        sys.exit(1)

    live_app = LiveInferApp(pipeline=pipeline, initial_params=initial_params)
    live_app.start()

    api = await InferAPI.serve(port, live_app)

    try:
        await asyncio.Event().wait()
    finally:
        await api.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        os._exit(0)
    except Exception:
        logging.exception("Fatal error in InferAPI main")
        os._exit(1)
import argparse
import asyncio
import json
import logging
import signal
import sys
import os
import traceback
import threading
from typing import List

from .process import ProcessGuardian
from .streamer import PipelineStreamer
from .streamer.protocol import TrickleProtocol, ZeroMQProtocol
from .trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT
from .api import start_http_server
from .log import config_logging, log_timing


_UNCAUGHT_EXCEPTION_EVENT = asyncio.Event()
_MAIN_LOOP: asyncio.AbstractEventLoop | None = None


def asyncio_exception_handler(loop, context):
    """
    Handles unhandled exceptions in asyncio tasks, logging the error and terminating the application.
    """
    exception = context.get('exception')
    logging.error(
        f"Terminating process due to unhandled exception in asyncio task: {exception}",
        exc_info=exception,
    )
    try:
        _UNCAUGHT_EXCEPTION_EVENT.set()
    except Exception:
        os._exit(1)


def thread_exception_hook(original_hook):
    """
    Creates a custom exception hook for threads that logs the error and terminates the application.
    """
    def custom_hook(args):
        logging.error(
            f"Terminating process due to unhandled exception in thread: {args.exc_value}",
            exc_info=args.exc_value,
        )
        original_hook(args)  # this is most likely a noop
        try:
            _MAIN_LOOP.call_soon_threadsafe(_UNCAUGHT_EXCEPTION_EVENT.set)  # type: ignore
        except Exception:
            os._exit(1)
    return custom_hook


async def main(
    *,
    http_port: int,
    stream_protocol: str,
    subscribe_url: str,
    publish_url: str,
    control_url: str,
    events_url: str,
    pipeline: str,
    params: dict,
    request_id: str,
    manifest_id: str,
    stream_id: str,
):
    global _MAIN_LOOP
    _MAIN_LOOP = asyncio.get_event_loop()
    _MAIN_LOOP.set_exception_handler(asyncio_exception_handler)

    process = ProcessGuardian(pipeline, params or {})
    # Only initialize the streamer if we have a protocol and URLs to connect to
    streamer = None
    if stream_protocol and subscribe_url and publish_url:
        width = params.get('width', DEFAULT_WIDTH)
        height = params.get('height', DEFAULT_HEIGHT)
        if stream_protocol == "trickle":
            protocol = TrickleProtocol(
                subscribe_url, publish_url, control_url, events_url, width, height
            )
        elif stream_protocol == "zeromq":
            protocol = ZeroMQProtocol(subscribe_url, publish_url)
        else:
            raise ValueError(f"Unsupported protocol: {stream_protocol}")
        streamer = PipelineStreamer(
            protocol, process, request_id, manifest_id, stream_id
        )

    api = None
    try:
        with log_timing("starting ProcessGuardian"):
            await process.start()
            if streamer:
                await streamer.start(params)
            api = await start_http_server(http_port, process, streamer)

        lifecycle_tasks: List[asyncio.Task] = [
            asyncio.create_task(block_until_signal([signal.SIGINT, signal.SIGTERM])),
            asyncio.create_task(wait_uncaught_exception()),
        ]
        if streamer:
            lifecycle_tasks.append(asyncio.create_task(streamer.wait()))

        await asyncio.wait(lifecycle_tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e
    finally:
        stop_coros: List[asyncio._CoroutineLike] = [
            process.stop(),
        ]
        if streamer:
            streamer.trigger_stop_stream()
            stop_coros.append(streamer.wait())
        if api:
            stop_coros.append(api.cleanup())

        try:
            stops = asyncio.gather(*stop_coros, return_exceptions=True)
            results = await asyncio.wait_for(stops, timeout=10)
            exceptions = [result for result in results if isinstance(result, Exception)]
            if exceptions:
                raise ExceptionGroup("Error stopping components", exceptions)
        except Exception as e:
            logging.error(f"Graceful shutdown error, exiting abruptly: {e}", exc_info=True)
            os._exit(1)


async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    def signal_handler(sig, _):
        logging.info(f"Received signal, initiating graceful shutdown. signal={sig}")
        loop.call_soon_threadsafe(future.set_result, sig)

    for sig in sigs:
        signal.signal(sig, signal_handler)
    return await future

async def wait_uncaught_exception():
    await _UNCAUGHT_EXCEPTION_EVENT.wait()
    logging.error(
        "Uncaught exception event received, initiating graceful shutdown."
    )

if __name__ == "__main__":
    threading.excepthook = thread_exception_hook(threading.excepthook)

    parser = argparse.ArgumentParser(description="Infer process to run the AI pipeline")
    parser.add_argument(
        "--http-port", type=int, default=8888, help="Port for the HTTP server"
    )
    parser.add_argument(
        "--pipeline", type=str, default="comfyui", help="Pipeline to use"
    )
    parser.add_argument(
        "--initial-params",
        type=str,
        default="{}",
        help="Initial parameters for the pipeline",
    )
    parser.add_argument(
        "--stream-protocol",
        type=str,
        choices=["trickle", "zeromq"],
        default=os.getenv("STREAM_PROTOCOL", "trickle"),
        help="Protocol to use for streaming frames in and out. One of: trickle, zeromq",
    )
    parser.add_argument(
        "--subscribe-url",
        type=str,
        help="URL to subscribe for the input frames (trickle). For zeromq this is the input socket address",
    )
    parser.add_argument(
        "--publish-url",
        type=str,
        help="URL to publish output frames (trickle). For zeromq this is the output socket address",
    )
    parser.add_argument(
        "--control-url",
        type=str,
        help="URL to subscribe for Control API JSON messages to update inference params",
    )
    parser.add_argument(
        "--events-url",
        type=str,
        help="URL to publish events about pipeline status and logs.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (debug) logging"
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default="",
        help="The Livepeer request ID associated with this video stream",
    )
    parser.add_argument(
        "--manifest-id", type=str, default="", help="The orchestrator manifest ID"
    )
    parser.add_argument(
        "--stream-id", type=str, default="", help="The Livepeer stream ID"
    )
    args = parser.parse_args()
    try:
        params = json.loads(args.initial_params)
    except Exception as e:
        logging.error(f"Error parsing --initial-params: {e}")
        sys.exit(1)

    if args.verbose:
        os.environ["VERBOSE_LOGGING"] = "1"  # enable verbose logging in sub-processes

    config_logging(
        log_level=logging.DEBUG if os.getenv("VERBOSE_LOGGING")=="1" else logging.INFO,
        request_id=args.request_id,
        manifest_id=args.manifest_id,
        stream_id=args.stream_id,
    )

    try:
        asyncio.run(
            main(
                http_port=args.http_port,
                stream_protocol=args.stream_protocol,
                subscribe_url=args.subscribe_url,
                publish_url=args.publish_url,
                control_url=args.control_url,
                events_url=args.events_url,
                pipeline=args.pipeline,
                params=params,
                request_id=args.request_id,
                manifest_id=args.manifest_id,
                stream_id=args.stream_id,
            )
        )
        # We force an exit here to ensure that the process terminates. If any asyncio tasks or
        # sub-processes failed to shutdown they'd block the main process from exiting.
        os._exit(0)
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        os._exit(1)

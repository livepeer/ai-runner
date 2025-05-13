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

from streamer import PipelineStreamer, ProcessGuardian

# loads neighbouring modules with absolute paths
infer_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, infer_root)

from api import start_http_server
from log import config_logging, log_timing
from streamer.protocol.trickle import TrickleProtocol
from streamer.protocol.zeromq import ZeroMQProtocol


def asyncio_exception_handler(loop, context):
    """
    Handles unhandled exceptions in asyncio tasks, logging the error and terminating the application.
    """
    exception = context.get('exception')
    logging.error(f"Terminating process due to unhandled exception in asyncio task", exc_info=exception)
    os._exit(1)


def thread_exception_hook(original_hook):
    """
    Creates a custom exception hook for threads that logs the error and terminates the application.
    """
    def custom_hook(args):
        logging.error("Terminating process due to unhandled exception in thread", exc_info=args.exc_value)
        original_hook(args) # this is most likely a noop
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
    stream_id: str,
):
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)

    process = ProcessGuardian(pipeline, params or {})
    # Only initialize the streamer if we have a protocol and URLs to connect to
    streamer = None
    if stream_protocol and subscribe_url and publish_url:
        if stream_protocol == "trickle":
            protocol = TrickleProtocol(
                subscribe_url, publish_url, control_url, events_url
            )
        elif stream_protocol == "zeromq":
            protocol = ZeroMQProtocol(subscribe_url, publish_url)
        else:
            raise ValueError(f"Unsupported protocol: {stream_protocol}")
        streamer = PipelineStreamer(protocol, process, request_id, stream_id)

    api = None
    try:
        with log_timing("starting ProcessGuardian"):
            await process.start()
            if streamer:
                await streamer.start(params)
            api = await start_http_server(http_port, process, streamer)

        tasks: List[asyncio.Task] = []
        if streamer:
            tasks.append(asyncio.create_task(streamer.wait()))
        tasks.append(
            asyncio.create_task(block_until_signal([signal.SIGINT, signal.SIGTERM]))
        )

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e
    finally:
        if streamer:
            streamer.trigger_stop_stream()
            await streamer.wait(timeout=5)
        if api:
            await api.cleanup()
        await process.stop()


async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    def signal_handler(sig, _):
        logging.info(f"Received signal: {sig}")
        loop.call_soon_threadsafe(future.set_result, sig)

    for sig in sigs:
        signal.signal(sig, signal_handler)
    return await future


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

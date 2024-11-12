import argparse
import asyncio
import json
import logging
import signal
import sys
import os
import traceback
from typing import List

# loads neighbouring modules with absolute paths
infer_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, infer_root)

from params_api import start_http_server
from streamer.trickle import TrickleStreamer
from streamer.zeromq import ZeroMQStreamer


async def main(http_port: int, stream_protocol: str, subscribe_url: str, publish_url: str, pipeline: str, params: dict):
    if stream_protocol == "trickle":
        handler = TrickleStreamer(subscribe_url, publish_url, pipeline, **(params or {}))
    elif stream_protocol == "zeromq":
        handler = ZeroMQStreamer(subscribe_url, publish_url, pipeline, **(params or {}))
    else:
        raise ValueError(f"Unsupported protocol: {stream_protocol}")

    runner = None
    try:
        handler.start()
        runner = await start_http_server(handler, http_port)
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e

    await block_until_signal([signal.SIGINT, signal.SIGTERM])
    try:
        await runner.cleanup()
        await handler.stop()
    except Exception as e:
        logging.error(f"Error stopping room handler: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e


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
    parser = argparse.ArgumentParser(description="Infer process to run the AI pipeline")
    parser.add_argument(
        "--http-port", type=int, default=8888, help="Port for the HTTP server"
    )
    parser.add_argument(
        "--pipeline", type=str, default="streamdiffusion", help="Pipeline to use"
    )
    parser.add_argument(
        "--initial-params", type=str, default="{}", help="Initial parameters for the pipeline"
    )
    parser.add_argument(
        "--stream-protocol",
        type=str,
        choices=["trickle", "zeromq"],
        default="trickle",
        help="Protocol to use for streaming frames in and out. One of: trickle, zeromq"
    )
    parser.add_argument(
        "--subscribe-url", type=str, required=True, help="URL to subscribe for the input frames (trickle). For zeromq this is the input socket address"
    )
    parser.add_argument(
        "--publish-url", type=str, required=True, help="URL to publish output frames (trickle). For zeromq this is the output socket address"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    args = parser.parse_args()
    try:
        params = json.loads(args.initial_params)
    except Exception as e:
        logging.error(f"Error parsing --initial-params: {e}")
        sys.exit(1)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=log_level,
        datefmt='%Y-%m-%d %H:%M:%S')
    if args.verbose:
        os.environ['VERBOSE_LOGGING'] = '1' # enable verbose logging in subprocesses

    try:
        asyncio.run(
            main(args.http_port, args.stream_protocol, args.subscribe_url, args.publish_url, args.pipeline, params)
        )
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        sys.exit(1)


import argparse
import asyncio
import json
import logging
import signal
import sys
import os
import traceback
from typing import List
import logging

from streamer import PipelineStreamer
from trickle import TrickleSubscriber

# loads neighbouring modules with absolute paths
infer_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, infer_root)

from params_api import start_http_server
from streamer.protocol.trickle import TrickleProtocol
from streamer.protocol.zeromq import ZeroMQProtocol


async def main(http_port: int, stream_protocol: str, subscribe_url: str, publish_url: str, control_url: str, pipeline: str, params: dict, input_timeout: int):
    if stream_protocol == "trickle":
        protocol = TrickleProtocol(subscribe_url, publish_url)
    elif stream_protocol == "zeromq":
        protocol = ZeroMQProtocol(subscribe_url, publish_url)
    else:
        raise ValueError(f"Unsupported protocol: {stream_protocol}")

    streamer = PipelineStreamer(protocol, pipeline, input_timeout, params or {})

    runner = None
    try:
        await streamer.start()
        runner = await start_http_server(streamer, http_port)

        tasks: List[asyncio.Task] = []
        tasks.append(asyncio.create_task(streamer.wait()))
        tasks.append(asyncio.create_task(block_until_signal([signal.SIGINT, signal.SIGTERM])))
        if control_url is not None and control_url.strip() != "":
            tasks.append(asyncio.create_task(start_control_subscriber(streamer, control_url)))

        await asyncio.wait(tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
    except Exception as e:
        logging.error(f"Error starting socket handler or HTTP server: {e}")
        logging.error(f"Stack trace:\n{traceback.format_exc()}")
        raise e
    finally:
        await runner.cleanup()
        await streamer.stop()


async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    def signal_handler(sig, _):
        logging.info(f"Received signal: {sig}")
        loop.call_soon_threadsafe(future.set_result, sig)

    for sig in sigs:
        signal.signal(sig, signal_handler)
    return await future

async def start_control_subscriber(handler: PipelineStreamer, control_url: str):
    if control_url is None or control_url.strip() == "":
        logging.warning("No control-url provided, inference won't get updates from the control trickle subscription")
        return
    logging.info("Starting Control subscriber at %s", control_url)
    subscriber = TrickleSubscriber(url=control_url)
    while True:
        segment = await subscriber.next()
        if segment.eos():
            return

        try:
            params = await segment.read()
            logging.info("Received control message, updating model with params: %s", params)
            data = json.loads(params)
        except Exception as e:
            logging.error(f"Error parsing control message: {e}")
            continue

        try:
            handler.update_params(data)
        except Exception as e:
            logging.error(f"Error updating model with control message: {e}")
            continue

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
        "--control-url", type=str, help="URL to subscribe for Control API JSON messages to update inference params"
    )
    parser.add_argument(
        "--input-timeout",
        type=int,
        default=60,
        help="Timeout in seconds to wait after input frames stop before shutting down. Set to 0 to disable."
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
            main(args.http_port, args.stream_protocol, args.subscribe_url, args.publish_url, args.control_url, args.pipeline, params, args.input_timeout)
        )
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        sys.exit(1)


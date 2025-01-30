import argparse
import asyncio
import json
import signal
import sys
import os
import traceback
from typing import List, Optional
import structlog

from streamer import PipelineStreamer

# loads neighbouring modules with absolute paths
infer_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, infer_root)

from api import start_http_server
from streamer.protocol.trickle import TrickleProtocol
from streamer.protocol.zeromq import ZeroMQProtocol


def setup_logging(stream_id: Optional[str] = None, trickle_id: Optional[str] = None) -> structlog.BoundLogger:
    """Setup structured logging with stream ID and trickle ID context"""
    
    # Configure structlog to output logfmt
    structlog.configure(
        processors=[
            # Add timestamps
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f"),
            # Add log level
            structlog.processors.add_log_level,
            # Add caller info (file, line number)
            structlog.processors.CallsiteParameterAdder(
                parameters={"filename", "lineno"}
            ),
            # Convert to logfmt
            structlog.processors.LogfmtRenderer()
        ],
        # Output to stderr to match current [infer.py] prefix behavior
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        # Cache logger
        cache_logger_on_first_use=True,
    )
    
    # Create a logger with default context
    logger = structlog.get_logger(
        component="infer",
    )
    
    # Bind IDs if available
    if stream_id:
        logger = logger.bind(stream_id=stream_id)
    if trickle_id:
        logger = logger.bind(trickle_id=trickle_id)
    
    return logger


def extract_trickle_id(url: str) -> Optional[str]:
    """Extract trickle ID from Trickle URL.
    Example URL format: https://172.17.0.1:8888/ai/trickle/{trickle_id}-out/230
    """
    try:
        # Split on trickle/ and take the next part
        parts = url.split('trickle/')
        if len(parts) > 1:
            # Take everything before the -out or -in and after the last /
            stream_part = parts[1].split('/')[0]
            return stream_part.split('-')[0]
    except Exception:
        return None
    return None


async def main(*, http_port: int, stream_protocol: str, subscribe_url: str, 
               publish_url: str, control_url: str, events_url: str, 
               pipeline: str, params: dict, input_timeout: int, stream_id: str):
    
    # Extract trickle ID from any of the URLs
    trickle_id = (
        extract_trickle_id(subscribe_url) or 
        extract_trickle_id(publish_url) or 
        extract_trickle_id(control_url) or 
        extract_trickle_id(events_url)
    )
    
    # Setup logging with both IDs
    log = setup_logging(stream_id=stream_id, trickle_id=trickle_id)
    
    if stream_protocol == "trickle":
        protocol = TrickleProtocol(subscribe_url, publish_url, control_url, events_url)
        log.info("protocol.init",
                 protocol="trickle",
                 subscribe_url=subscribe_url,
                 publish_url=publish_url,
                 control_url=control_url,
                 events_url=events_url)
    elif stream_protocol == "zeromq":
        if events_url:
            log.warning("protocol.zeromq.no_events")
        if control_url:
            log.warning("protocol.zeromq.no_control")
        protocol = ZeroMQProtocol(subscribe_url, publish_url)
        log.info("protocol.init",
                 protocol="zeromq",
                 subscribe_url=subscribe_url,
                 publish_url=publish_url)
    else:
        log.error("protocol.unsupported", protocol=stream_protocol)
        raise ValueError(f"Unsupported protocol: {stream_protocol}")

    streamer = PipelineStreamer(protocol, pipeline, input_timeout, params or {})

    api = None
    try:
        await streamer.start()
        log.info("streamer.start", pipeline=pipeline)
        
        api = await start_http_server(http_port, streamer)
        log.info("http.start", port=http_port)

        tasks: List[asyncio.Task] = []
        tasks.append(streamer.wait())
        tasks.append(asyncio.create_task(
            block_until_signal([signal.SIGINT, signal.SIGTERM]))
        )

        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except Exception as e:
        log.error("stream.error",
                  error=str(e),
                  error_type=type(e).__name__,
                  traceback=traceback.format_exc())
        raise e
    finally:
        await streamer.stop(timeout=5)
        log.info("streamer.stop")
        if api:
            await api.cleanup()
            log.info("api.cleanup")


async def block_until_signal(sigs: List[signal.Signals]):
    loop = asyncio.get_running_loop()
    future: asyncio.Future[signal.Signals] = loop.create_future()

    # Get the logger - it will have trickle_id if it was set in main()
    log = structlog.get_logger()

    def signal_handler(sig, _):
        log.info("signal.received", signal=sig)
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
        default=os.getenv("STREAM_PROTOCOL", "trickle"),
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
        "--events-url", type=str, help="URL to publish events about pipeline status and logs."
    )
    parser.add_argument(
        "--input-timeout",
        type=int,
        default=60,
        help="Timeout in seconds to wait after input frames stop before shutting down. Set to 0 to disable."
    )
    parser.add_argument(
        "--stream-id",
        type=str,
        default="",
        help="The Livepeer stream ID associated with this video stream"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    args = parser.parse_args()

    # Setup initial logger without IDs
    log = setup_logging()
    
    try:
        params = json.loads(args.initial_params)
    except Exception as e:
        log.error("params.parse_error", error=str(e))
        sys.exit(1)

    if args.verbose:
        os.environ['VERBOSE_LOGGING'] = '1' # enable verbose logging in subprocesses

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
                input_timeout=args.input_timeout,
                stream_id=args.stream_id
            )
        )
        # We force an exit here to ensure that the process terminates. If any asyncio tasks or
        # sub-processes failed to shutdown they'd block the main process from exiting.
        os._exit(0)
    except Exception as e:
        log.error("main.fatal_error",
                  error=str(e),
                  error_type=type(e).__name__,
                  traceback=''.join(traceback.format_tb(e.__traceback__)))
        os._exit(1)


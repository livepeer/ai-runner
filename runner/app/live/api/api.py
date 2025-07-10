import asyncio
import logging
import json
import os
import tempfile
import time
from typing import Optional, cast

from aiohttp import web
from pydantic import BaseModel, Field
from typing import Annotated, Dict

from streamer import PipelineStreamer, ProcessGuardian
from streamer.protocol.trickle import TrickleProtocol
from streamer.process import config_logging
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT

MAX_FILE_AGE = 86400  # 1 day

# File to store the last params that a stream was started with. Used to cleanup
# left over resources (e.g. trickle channels) left by a crashed process.
last_params_file = os.path.join(tempfile.gettempdir(), "ai_runner_last_params.json")

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

async def cleanup_last_stream():
    if not os.path.exists(last_params_file):
        logging.debug("No last stream params found to cleanup")
        return

    try:
        with open(last_params_file, "r") as f:
            params = StartStreamParams(**json.load(f))
        os.remove(last_params_file)

        logging.info(f"Cleaning up last stream trickle channels for request_id={params.request_id} subscribe_url={params.subscribe_url} publish_url={params.publish_url} control_url={params.control_url} events_url={params.events_url}")
        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
        )
        # Start and stop the protocol to immediately to make sure trickle channels are closed.
        await protocol.start()
        await protocol.stop()
    except:
        logging.exception(f"Error cleaning up last stream trickle channels")

async def parse_request_data(request: web.Request) -> Dict:
    if request.content_type.startswith("application/json"):
        return await request.json()
    else:
        raise ValueError(f"Unknown content type: {request.content_type}")

async def handle_start_stream(request: web.Request):
    try:
        stream_request_timestamp = int(time.time() * 1000)
        process = cast(ProcessGuardian, request.app["process"])
        prev_streamer = cast(PipelineStreamer, request.app["streamer"])
        if prev_streamer and prev_streamer.is_running():
            # Stop the previous streamer before starting a new one
            try:
                logging.info("Stopping previous streamer")
                prev_streamer.trigger_stop_stream()
                await prev_streamer.wait(timeout=10)
            except asyncio.TimeoutError as e:
                logging.error(f"Timeout stopping streamer: {e}")
                raise web.HTTPBadRequest(text="Timeout stopping previous streamer")

        params_data = await parse_request_data(request)
        params = StartStreamParams(**params_data)

        try:
            with open(last_params_file, "w") as f:
                json.dump(params.model_dump(), f)
        except Exception as e:
            logging.error(f"Error saving last params to file: {e}")

        config_logging(request_id=params.request_id, manifest_id=params.manifest_id, stream_id=params.stream_id)

        # Try to get dimensions from workflow first
        width = params.params.get("width", DEFAULT_WIDTH)
        height = params.params.get("height", DEFAULT_HEIGHT)
        if process.pipeline == "comfyui":
            # TODO: Remove this once ComfyUI pipeline supports different resolutions without a restart
            width = height = 512
            params.params = params.params | {"width": width, "height": height}
            logging.warning("Using default dimensions for ComfyUI pipeline")
        else:
            logging.info(f"Using dimensions from params: {width}x{height}")

        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
            width,
            height,
        )
        streamer = PipelineStreamer(
            protocol,
            process,
            params.request_id,
            params.manifest_id,
            params.stream_id,
        )

        await streamer.start(params.params)
        request.app["streamer"] = streamer
        await protocol.emit_monitoring_event({
            "type": "runner_receive_stream_request",
            "timestamp": stream_request_timestamp,
        }, queue_event_type="stream_trace")

        return web.Response(text="Stream started successfully")
    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        return web.Response(text=f"Error starting stream: {str(e)}", status=400)


async def handle_params_update(request: web.Request):
    try:
        params = await parse_request_data(request)

        process = cast(ProcessGuardian, request.app["process"])
        await process.update_params(params)

        return web.Response(text="Params updated successfully")
    except Exception as e:
        logging.error(f"Error updating params: {e}")
        return web.Response(text=f"Error updating params: {str(e)}", status=400)


async def handle_get_status(request: web.Request):
    process = cast(ProcessGuardian, request.app["process"])
    status = process.get_status()
    return web.json_response(status.model_dump())


async def start_http_server(
    port: int, process: ProcessGuardian, streamer: Optional[PipelineStreamer] = None
):
    asyncio.create_task(cleanup_last_stream())

    app = web.Application()
    app["process"] = process
    app["streamer"] = streamer
    app.router.add_post("/api/live-video-to-video", handle_start_stream)
    app.router.add_post("/api/params", handle_params_update)
    app.router.add_get("/api/status", handle_get_status)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"HTTP server started on port {port}")
    return runner

import asyncio
import hashlib
import logging
import mimetypes
import os
import tempfile
import time
from typing import Optional, cast

from aiohttp import BodyPartReader, web
from pydantic import BaseModel, Field
from typing import Annotated, Dict

from streamer import PipelineStreamer, ProcessGuardian
from streamer.protocol.trickle import TrickleProtocol
from streamer.process import config_logging

TEMP_SUBDIR = "infer_temp"
MAX_FILE_AGE = 86400  # 1 day
STREAMER_INPUT_TIMEOUT = 60  # 60s


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
    stream_id: Annotated[
        str,
        Field(default="", description="Unique identifier for the stream."),
    ]


def cleanup_old_files(temp_dir):
    current_time = time.time()
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > MAX_FILE_AGE:
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path}")


async def parse_request_data(request: web.Request, temp_dir: str) -> Dict:
    if request.content_type.startswith("application/json"):
        return await request.json()
    elif request.content_type.startswith("multipart/"):
        params_data = {}
        reader = await request.multipart()
        async for part in reader:
            if not isinstance(part, BodyPartReader):
                continue
            elif part.name == "params":
                part_data = await part.json()
                if part_data:
                    params_data.update(part_data)
            else:
                content = await part.read()
                file_hash = hashlib.md5(content).hexdigest()
                content_type = part.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                ext = mimetypes.guess_extension(content_type) or ""
                new_filename = f"{file_hash}{ext}"
                file_path = os.path.join(temp_dir, new_filename)
                with open(file_path, "wb") as f:
                    f.write(content)
                params_data[part.name] = file_path
        return params_data
    else:
        raise ValueError(f"Unknown content type: {request.content_type}")


async def handle_start_stream(request: web.Request):
    try:
        process = cast(ProcessGuardian, request.app["process"])
        prev_streamer = cast(PipelineStreamer, request.app["streamer"])
        if prev_streamer and prev_streamer.is_running():
            # Stop the previous streamer before starting a new one
            try:
                logging.info("Stopping previous streamer")
                await prev_streamer.stop(timeout=10)
            except asyncio.TimeoutError as e:
                logging.error(f"Timeout stopping streamer: {e}")
                raise web.HTTPBadRequest(text="Timeout stopping previous streamer")

        temp_dir = os.path.join(tempfile.gettempdir(), TEMP_SUBDIR)
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_old_files(temp_dir)

        params_data = await parse_request_data(request, temp_dir)
        params = StartStreamParams(**params_data)

        config_logging(request_id=params.request_id, stream_id=params.stream_id)

        protocol = TrickleProtocol(
            params.subscribe_url,
            params.publish_url,
            params.control_url,
            params.events_url,
        )
        streamer = PipelineStreamer(
            protocol,
            STREAMER_INPUT_TIMEOUT,
            process,
            params.request_id,
            params.stream_id,
        )

        await streamer.start(params.params)
        request.app["streamer"] = streamer

        return web.Response(text="Stream started successfully")
    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        return web.Response(text=f"Error starting stream: {str(e)}", status=400)


async def handle_params_update(request: web.Request):
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), TEMP_SUBDIR)
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_old_files(temp_dir)

        params = await parse_request_data(request, temp_dir)

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

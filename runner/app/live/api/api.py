import asyncio
import logging
import json
from typing import cast

from aiohttp import web
from pydantic import BaseModel, Field
from typing import Annotated, Dict

from ..log import config_logging
from ..live_infer_app import LiveInferApp

MAX_FILE_AGE = 86400  # 1 day

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

async def parse_request_data(request: web.Request) -> Dict:
    if request.content_type.startswith("application/json"):
        return await request.json()
    else:
        raise ValueError(f"Unknown content type: {request.content_type}")

async def handle_start_stream(request: web.Request):
    try:
        params_data = await parse_request_data(request)
        params = StartStreamParams(**params_data)

        live_app = cast(LiveInferApp, request.app["live_infer_app"])
        config_logging(request_id=params.request_id, manifest_id=params.manifest_id, stream_id=params.stream_id)

        await live_app.a_start_stream(
            subscribe_url=params.subscribe_url,
            publish_url=params.publish_url,
            control_url=params.control_url,
            events_url=params.events_url,
            params=params.params,
            request_id=params.request_id,
            manifest_id=params.manifest_id,
            stream_id=params.stream_id,
        )

        return web.Response(text="Stream started successfully")
    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        return web.Response(text=f"Error starting stream: {str(e)}", status=400)


async def handle_params_update(request: web.Request):
    try:
        params = await parse_request_data(request)

        live_app = cast(LiveInferApp, request.app["live_infer_app"])
        await live_app.a_update_params(params)

        return web.Response(text="Params updated successfully")
    except Exception as e:
        logging.error(f"Error updating params: {e}")
        return web.Response(text=f"Error updating params: {str(e)}", status=400)


async def handle_get_status(request: web.Request):
    live_app = cast(LiveInferApp, request.app["live_infer_app"])
    status = await live_app.a_get_status()
    return web.json_response(status.model_dump())


async def start_http_server(port: int, live_infer_app: LiveInferApp):
    app = web.Application()
    app["live_infer_app"] = live_infer_app
    app.router.add_post("/api/live-video-to-video", handle_start_stream)
    app.router.add_post("/api/params", handle_params_update)
    app.router.add_get("/api/status", handle_get_status)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"HTTP server started on port {port}")
    return runner

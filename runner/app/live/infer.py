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
from typing import Optional

from aiohttp import web

from .live_infer_app import LiveInferApp, StreamParams


class InferAPI:
    @classmethod
    async def serve(cls, port: int, live_infer_app: LiveInferApp):
        api = cls(live_infer_app)
        api.runner = web.AppRunner(api.app)
        await api.runner.setup()
        api.site = web.TCPSite(api.runner, "0.0.0.0", port)
        await api.site.start()
        logging.info(f"HTTP server started on port {port}")
        return api

    def __init__(self, live_infer_app: LiveInferApp):
        self.live_infer_app = live_infer_app
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Routes
        self.app.router.add_post("/api/live-video-to-video", self.handle_start_stream)
        self.app.router.add_post("/api/params", self.handle_params_update)
        self.app.router.add_get("/api/status", self.handle_get_status)

    async def handle_start_stream(self, request: web.Request):
        try:
            params_data = await request.json()
            params = StreamParams(**params_data)

            await self.live_infer_app.start_stream(params)
            return web.Response(text="Stream started successfully")
        except Exception as e:
            logging.error(f"Error starting stream: {e}")
            return web.Response(text=f"Error starting stream: {str(e)}", status=500)

    async def handle_params_update(self, request: web.Request):
        try:
            params = await request.json()
            await self.live_infer_app.update_params(params)
            return web.Response(text="Params updated successfully")
        except Exception as e:
            logging.error(f"Error updating params: {e}")
            return web.Response(text=f"Error updating params: {str(e)}", status=500)

    async def handle_get_status(self, request: web.Request):
        status = await self.live_infer_app.get_status()
        return web.json_response(status.model_dump())

    async def shutdown(self):
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
            self.site = None


async def main():
    port = int(os.environ.get("INFER_HTTP_PORT", "8888"))
    pipeline = os.environ.get("PIPELINE", "streamdiffusion")
    initial_params_env = os.environ.get("INFERPY_INITIAL_PARAMS", "{}")
    try:
        initial_params = json.loads(initial_params_env) if initial_params_env else {}
    except Exception as e:
        logging.error(f"Error parsing INFERPY_INITIAL_PARAMS: {e}")
        sys.exit(1)

    live_app = LiveInferApp(pipeline=pipeline, initial_params=initial_params)
    live_app.setup_fatal_signal_handlers()
    await live_app.start()

    api = await InferAPI.serve(port, live_app)

    try:
        await live_app.wait_for_stop()
    finally:
        await api.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        os._exit(0)
    except Exception:
        logging.exception("Fatal error in InferAPI main")
        os._exit(1)

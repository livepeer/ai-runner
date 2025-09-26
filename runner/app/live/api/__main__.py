import asyncio
import json
import logging
import os
import sys

from .api import InferAPI, start_http_server
from ..live_infer_app import LiveInferApp


async def main():
    port = int(os.environ.get("INFER_HTTP_PORT", "8888"))
    pipeline = os.environ.get("PIPELINE", "comfyui")
    initial_params_env = os.environ.get("INFERPY_INITIAL_PARAMS", "{}")
    try:
        initial_params = json.loads(initial_params_env) if initial_params_env else {}
    except Exception as e:
        logging.error(f"Error parsing INFERPY_INITIAL_PARAMS: {e}")
        sys.exit(1)

    app = LiveInferApp(pipeline=pipeline, initial_params=initial_params)
    app.start()

    infer_api = InferAPI(app)
    runner = await start_http_server(port, infer_api)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        os._exit(0)
    except Exception:
        logging.exception("Fatal error in InferAPI main")
        os._exit(1)


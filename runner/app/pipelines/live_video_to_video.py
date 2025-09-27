import asyncio
import json
import logging
import os
import threading
from typing import Callable, Coroutine, Any, TypeVar

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
from app.utils.errors import InferenceError
from app.live import StreamParams, SyncLiveInferApp


class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = get_model_dir()
        self.torch_device = get_torch_device()

        initial_params_env = os.environ.get("INFERPY_INITIAL_PARAMS")
        try:
            initial_params = json.loads(initial_params_env) if initial_params_env else {}
        except Exception as e:
            logging.error(f"Error parsing INFERPY_INITIAL_PARAMS: {e}")
            initial_params = {}

        self.app = SyncLiveInferApp(self.model_id, initial_params)
        self.app.setup_fatal_signal_handlers()
        self.app.start()

    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, stream_params: dict, request_id: str, manifest_id: str, stream_id: str, **kwargs
    ):
        try:
            self.app._run(lambda app: app.start_stream(StreamParams(
                subscribe_url=subscribe_url,
                publish_url=publish_url,
                control_url=control_url,
                events_url=events_url,
                params=stream_params,
                request_id=request_id or "",
                manifest_id=manifest_id or "",
                stream_id=stream_id or "",
            )))
            logging.info("Stream started successfully")
            return {}
        except Exception as e:
            logging.error("Failed to start stream", exc_info=True)
            raise InferenceError(original_exception=e)

    def get_health(self) -> HealthCheck:
        try:
            state = self.app._run(lambda app: app.get_status()).state
            return HealthCheck(
                status=(
                    "LOADING" if state == "LOADING"
                    else "IDLE" if state == "OFFLINE"
                    else "ERROR" if state == "ERROR"
                    else "OK"
                ),
            )
        except Exception as e:
            msg = "Failed to get live pipeline status"
            logging.error(f"[HEALTHCHECK] {msg}", exc_info=True)
            raise ConnectionError(f"{msg}: {e}") from e

    def __str__(self) -> str:
        return f"LiveVideoToVideoPipeline model_id={self.model_id}"



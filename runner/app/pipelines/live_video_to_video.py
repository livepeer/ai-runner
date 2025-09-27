import json
import logging
import os

from app.pipelines.base import Pipeline, HealthCheck
from app.pipelines.utils import get_model_dir, get_torch_device
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

        self.app = SyncLiveInferApp(self.model_id, initial_params, str(self.model_dir))
        self.app.setup_fatal_signal_handlers()
        self.app.start()

    def __call__(  # type: ignore
        self, *, subscribe_url: str, publish_url: str, control_url: str, events_url: str, params: dict, request_id: str, manifest_id: str, stream_id: str, **kwargs
    ):
        self.app.start_stream(StreamParams(
            subscribe_url=subscribe_url,
            publish_url=publish_url,
            control_url=control_url,
            events_url=events_url,
            params=params,
            request_id=request_id or "",
            manifest_id=manifest_id or "",
            stream_id=stream_id or "",
        ))
        logging.info("Stream started successfully")
        return {}

    def get_health(self) -> HealthCheck:
        state = self.app.get_status().state
        # TODO: Merge the pipeline healthcheck with the live pipeline states to avoid this mapping
        return HealthCheck(
            status=(
                "LOADING" if state == "LOADING"
                else "IDLE" if state == "OFFLINE"
                else "ERROR" if state == "ERROR"
                else "OK"
            ),
        )

    def __str__(self) -> str:
        return f"LiveVideoToVideoPipeline model_id={self.model_id}"



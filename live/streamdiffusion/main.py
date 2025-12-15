import json
import os

from app.app import start_app
from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
from app.live.pipelines import PipelineSpec

initial_params = json.loads(os.environ.get("PIPE_INITIAL_PARAMS", "{}"))

pipeline_spec = PipelineSpec(
    name="streamdiffusion",
    pipeline_cls="pipeline.pipeline:StreamDiffusion",
    params_cls="pipeline.params:StreamDiffusionParams",
    initial_params=initial_params,
)

if __name__ == "__main__":
    start_app(pipeline=LiveVideoToVideoPipeline(pipeline_spec))


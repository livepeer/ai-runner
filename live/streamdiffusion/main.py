from app.app import create_app, start_app
from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
from app.live.pipelines import PipelineSpec

pipeline_spec = PipelineSpec(
    name="streamdiffusion",
    pipeline_cls="pipeline.pipeline:StreamDiffusion",
    params_cls="pipeline.params:StreamDiffusionParams",
)

if __name__ == "__main__":
    start_app(pipeline=LiveVideoToVideoPipeline(pipeline_spec))


from app.app import start_app
from app.pipelines.live_video_to_video import LiveVideoToVideoPipeline
from app.live.pipelines import PipelineSpec

pipeline_spec = PipelineSpec(
    name="scope",
    pipeline_cls="pipeline.pipeline:Scope",
    params_cls="pipeline.params:ScopeParams",
)

if __name__ == "__main__":
    start_app(pipeline=LiveVideoToVideoPipeline(pipeline_spec))


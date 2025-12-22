from runner.app import start_app
from runner.pipelines.live_video_to_video import LiveVideoToVideoPipeline
from runner.live.pipelines import PipelineSpec

pipeline_spec = PipelineSpec(
    name="comfyui",
    pipeline_cls="pipeline.pipeline:ComfyUI",
    params_cls="pipeline.params:ComfyUIParams",
)

if __name__ == "__main__":
    start_app(pipeline=LiveVideoToVideoPipeline(pipeline_spec))


from runner.app import start_app
from runner.live.pipelines import PipelineSpec

pipeline_spec = PipelineSpec(
    name="scope",
    pipeline_cls="pipeline.pipeline:Scope",
    params_cls="pipeline.params:ScopeParams",
)

if __name__ == "__main__":
    start_app(pipeline=pipeline_spec)


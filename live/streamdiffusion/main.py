import json
import os
from pathlib import Path

from runner.app import start_app
from runner.live.pipelines import PipelineSpec

initial_params = {}

subvariant = os.environ.get("SUBVARIANT")
if subvariant:
    params_filename = subvariant.replace("-", "_") + ".json"
    params_path = Path(__file__).parent / "default_params" / params_filename

    with open(params_path) as f:
        initial_params = json.load(f)

# Use subvariant as suffix unless it's sdturbo (default)
name_suffix = "" if subvariant == "sdturbo" else f"-{subvariant}"

pipeline_spec = PipelineSpec(
    name=f"streamdiffusion{name_suffix}",
    pipeline_cls="pipeline.pipeline:StreamDiffusion",
    params_cls="pipeline.params:StreamDiffusionParams",
    initial_params=initial_params,
)

if __name__ == "__main__":
    start_app(pipeline=pipeline_spec)


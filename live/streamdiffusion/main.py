import json
import os
from pathlib import Path

from runner.app import start_app
from runner.live.pipelines import PipelineSpec

initial_params = {}

# SUBVARIANT controls default params. Empty defaults to sdturbo params.
subvariant = os.environ.get("SUBVARIANT", "")
params_subvariant = subvariant or "sdturbo"  # Empty SUBVARIANT uses sdturbo params
params_filename = params_subvariant.replace("-", "_") + ".json"
params_path = Path(__file__).parent / "default_params" / params_filename

with open(params_path) as f:
    initial_params = json.load(f)

# Use subvariant as suffix unless it's empty
name_suffix = "" if not subvariant else f"-{subvariant}"

pipeline_spec = PipelineSpec(
    name=f"streamdiffusion{name_suffix}",
    pipeline_cls="pipeline.pipeline:StreamDiffusion",
    params_cls="pipeline.params:StreamDiffusionParams",
    initial_params=initial_params,
)

if __name__ == "__main__":
    start_app(pipeline=pipeline_spec)


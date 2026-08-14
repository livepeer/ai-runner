"""Run an example pipeline as a live server you can interact with.

Usage:
    cd runner
    PYTHONPATH=src:../examples/live-video-to-video uv run python \
        ../examples/live-video-to-video/test_examples.py [green-shift|edge-detect]
"""

import sys

from runner.app import start_app
from runner.live.pipelines import PipelineSpec

EXAMPLES = {
    "green-shift": PipelineSpec(
        name="green-shift",
        pipeline_cls="green_shift:green_shift",
    ),
    "edge-detect": PipelineSpec(
        name="edge-detect",
        pipeline_cls="edge_detect:EdgeDetect",
        params_cls="edge_detect:EdgeParams",
    ),
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "green-shift"
    spec = EXAMPLES.get(name)
    if spec is None:
        print(f"Unknown example: {name}")
        print(f"Available: {', '.join(EXAMPLES)}")
        sys.exit(1)

    print(f"Starting {name} pipeline...")
    start_app(pipeline=spec)

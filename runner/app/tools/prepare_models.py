"""
Common entrypoint to prepare pipeline-specific models and TensorRT engines.

Usage (inside container):
    python -m app.tools.prepare_models --pipeline streamdiffusion

The models directory is configured via HUGGINGFACE_HUB_CACHE environment variable
(set in the Docker image).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..live.pipelines import builtin_pipeline_spec
from ..live.pipelines.loader import load_pipeline_class


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare models for a pipeline.")
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Pipeline built-in name or import path for loading the pipeline"
        "(e.g. streamdiffusion, scope, custom_pipeline.package:PipelineClass).",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        required=False,
        help="Custom pipeline name (e.g. streamdiffusion, scope, custom).",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    # Models directory is configured via HUGGINGFACE_HUB_CACHE env var (set in Docker image)
    models_dir = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/models"))
    if not models_dir.exists():
        raise ValueError(f"Models dir {models_dir} does not exist (check HUGGINGFACE_HUB_CACHE env var)")

    logging.info("Loading pipeline '%s' for model preparation", args.pipeline)

    pipeline_spec = builtin_pipeline_spec(args.pipeline)
    if pipeline_spec:
        pipeline_class = load_pipeline_class(pipeline_spec.pipeline_cls)
    else:
        # This can be called with the pipeline import path for custom pipelines.
        pipeline_class = load_pipeline_class(args.pipeline)

    # This will download and/or compile any assets required for this pipeline. Might take a while.
    pipeline_class.prepare_models()

    logging.info("Model preparation finished successfully.")


if __name__ == "__main__":
    main()


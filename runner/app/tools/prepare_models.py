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

from ..live.pipelines.loader import load_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare models for a pipeline.")
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline name accepted by loader.load_pipeline "
        "(e.g. streamdiffusion, scope, comfyui).",
    )
    parser.add_argument(
        "--verbose",
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
    pipeline = load_pipeline(args.pipeline)
    try:
        pipeline.prepare_models()
    finally:
        del pipeline
    logging.info("Model preparation finished successfully.")


if __name__ == "__main__":
    main()


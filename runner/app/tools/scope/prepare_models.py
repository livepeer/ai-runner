"""
Utility entrypoint to prepare Scope models inside the ai-runner container.

For now, this acts as a smoke test to ensure the Scope repository and its
dependencies have been installed properly by importing a lightweight module.
Future revisions can extend this script to download assets or compile
TensorRT engines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("scope.prepare_models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Scope models and verify runtime dependencies."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/models"),
        help="Directory where Scope assets should live (default: /models).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def verify_scope_installation() -> None:
    """
    Ensure Scope python dependencies are importable.

    This primarily verifies that `uv sync` completed successfully in the Docker
    image. Importing `HealthResponse` is lightweight and exercises the same
    module path that runtime code relies on.
    """

    try:
        from lib.schema import HealthResponse  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime smoke test
        LOG.error(
            "Failed to import Scope libraries. Make sure uv sync completed successfully."
        )
        raise RuntimeError("Scope Python dependencies are not installed.") from exc

    LOG.info(
        "Verified Scope installation via lib.schema.HealthResponse (class=%s)",
        HealthResponse.__qualname__,
    )


def ensure_models_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    LOG.info("Using models directory: %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    LOG.info("Starting Scope model preparation workflow.")
    ensure_models_dir(args.models_dir)
    verify_scope_installation()
    LOG.info("Scope model preparation complete.")


if __name__ == "__main__":
    main()



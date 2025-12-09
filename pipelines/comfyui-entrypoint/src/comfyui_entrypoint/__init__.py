"""ComfyUI pipeline entrypoint for AI Runner.

This module provides the CLI entrypoint that starts the AI Runner
with the ComfyUI pipeline configured.
"""

import os
import sys


# Pipeline import paths
PIPELINE_IMPORT = "app.live.pipelines.comfyui.pipeline:ComfyUI"
PARAMS_IMPORT = "app.live.pipelines.comfyui.params:ComfyUIParams"


def main():
    """Main entrypoint that starts uvicorn with the AI Runner app."""
    # Set environment variables for the live pipeline
    os.environ.setdefault("PIPELINE", "live-video-to-video")
    os.environ.setdefault("MODEL_ID", "comfyui")
    os.environ.setdefault("PIPELINE_IMPORT", PIPELINE_IMPORT)
    os.environ.setdefault("PARAMS_IMPORT", PARAMS_IMPORT)

    # Start uvicorn
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_config="app/cfg/uvicorn_logging_config.json",
    )


if __name__ == "__main__":
    main()

# ComfyUI Entrypoint

This package provides the entrypoint for running AI Runner with the ComfyUI pipeline.

## Installation

```bash
pip install -e .
```

## Usage

After installation, run:

```bash
comfyui-runner
```

This starts the AI Runner FastAPI server with ComfyUI configured as the live pipeline.

## Environment Variables

The entrypoint sets these environment variables (can be overridden):

- `PIPELINE=live-video-to-video`
- `MODEL_ID=comfyui`
- `PIPELINE_IMPORT=app.live.pipelines.comfyui.pipeline:ComfyUI`
- `PARAMS_IMPORT=app.live.pipelines.comfyui.params:ComfyUIParams`


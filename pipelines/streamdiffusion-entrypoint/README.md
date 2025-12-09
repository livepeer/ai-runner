# StreamDiffusion Entrypoint

This package provides the entrypoint for running AI Runner with the StreamDiffusion pipeline.

## Installation

```bash
pip install -e .
```

## Usage

After installation, run:

```bash
streamdiffusion-runner
```

This starts the AI Runner FastAPI server with StreamDiffusion configured as the live pipeline.

## Environment Variables

The entrypoint sets these environment variables (can be overridden):

- `PIPELINE=live-video-to-video`
- `MODEL_ID=streamdiffusion`
- `PIPELINE_IMPORT=app.live.pipelines.streamdiffusion.pipeline:StreamDiffusion`
- `PARAMS_IMPORT=app.live.pipelines.streamdiffusion.params:StreamDiffusionParams`


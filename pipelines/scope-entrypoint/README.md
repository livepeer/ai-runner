# Scope Entrypoint

This package provides the entrypoint for running AI Runner with the Scope pipeline.

## Installation

```bash
pip install -e .
```

## Usage

After installation, run:

```bash
scope-runner
```

This starts the AI Runner FastAPI server with Scope configured as the live pipeline.

## Environment Variables

The entrypoint sets these environment variables (can be overridden):

- `PIPELINE=live-video-to-video`
- `MODEL_ID=scope`
- `PIPELINE_IMPORT=app.live.pipelines.scope.pipeline:Scope`
- `PARAMS_IMPORT=app.live.pipelines.scope.params:ScopeParams`


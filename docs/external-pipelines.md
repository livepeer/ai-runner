# External Pipeline Development Guide

This guide explains how to create AI Runner pipelines that live in separate repositories and can be installed as dependencies.

## Overview

AI Runner supports external pipelines through import-based loading. This allows pipeline developers to:

1. **Maintain separate repositories** - Each pipeline can have its own repo
2. **Use ai-runner-base as a library** - Install `ai-runner-base` as a dependency
3. **Simple configuration** - Pipelines are loaded via import paths, no entry point registration needed
4. **Multi-processing support** - Works seamlessly with the multiprocessing architecture

## Architecture

### Multi-Processing Context

AI Runner uses `multiprocessing` with `spawn` context for GPU memory isolation. This means:

- Pipelines run in a **separate spawned process**
- The pipeline code must be **importable** from installed packages
- Import paths are passed to the spawned process via CLI arguments

### Import-Based Loading

Pipelines are loaded via explicit import paths:

- **`--pipeline-import`** - Full import path for Pipeline class (e.g., `app.live.pipelines.noop:Noop`)
- **`--params-import`** - Full import path for Params class (optional - empty uses `BaseParams`)

These are typically set via environment variables (`PIPELINE_IMPORT`, `PARAMS_IMPORT`) by the entrypoint package.

## Creating an External Pipeline

### Step 1: Project Structure

Create a new Python package with this structure:

```
my-pipeline/
├── pyproject.toml
├── README.md
├── src/
│   └── my_pipeline/
│       ├── __init__.py
│       ├── pipeline.py
│       └── params.py
└── my-pipeline-entrypoint/
    ├── pyproject.toml
    └── src/
        └── my_pipeline_entrypoint/
            └── __init__.py
```

### Step 2: Implement Pipeline Interface

Create `src/my_pipeline/pipeline.py`:

```python
from app.live.pipelines.interface import Pipeline
from app.live.trickle import VideoFrame, VideoOutput
import asyncio
import logging

class MyPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.initialized = False

    async def initialize(self, **params):
        """Initialize the pipeline with parameters."""
        logging.info(f"Initializing MyPipeline with params: {params}")
        # Your initialization logic here
        self.initialized = True

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        """Process an input frame."""
        # Your processing logic here
        # For example, a simple pass-through:
        output = VideoOutput(frame, request_id)
        await self.frame_queue.put(output)

    async def get_processed_video_frame(self) -> VideoOutput:
        """Get the next processed frame."""
        return await self.frame_queue.get()

    async def update_params(self, **params):
        """Update pipeline parameters."""
        logging.info(f"Updating params: {params}")
        # Return a Task if reload is needed, None otherwise
        return None

    async def stop(self):
        """Clean up resources."""
        self.frame_queue = asyncio.Queue()
        self.initialized = False

    @classmethod
    def prepare_models(cls):
        """Download/prepare models if needed."""
        logging.info("Preparing MyPipeline models")
        # Your model preparation logic
```

### Step 3: Implement Parameters

Create `src/my_pipeline/params.py`:

```python
from app.live.pipelines.interface import BaseParams
from pydantic import Field

class MyPipelineParams(BaseParams):
    """Parameters for MyPipeline."""

    # Add your custom parameters
    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Processing strength"
    )

    # BaseParams already provides: width, height, show_reloading_frame
```

### Step 4: Create Entrypoint Package

Create `my-pipeline-entrypoint/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-pipeline-entrypoint"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ai-runner-base>=0.1.0",
    # Your pipeline package if published separately
]

[project.scripts]
my-pipeline-runner = "my_pipeline_entrypoint:main"

[tool.setuptools]
packages = ["my_pipeline_entrypoint"]

[tool.setuptools.package-dir]
"" = "src"
```

Create `my-pipeline-entrypoint/src/my_pipeline_entrypoint/__init__.py`:

```python
"""My Pipeline entrypoint for AI Runner."""

import os

# Pipeline import paths
PIPELINE_IMPORT = "my_pipeline.pipeline:MyPipeline"
PARAMS_IMPORT = "my_pipeline.params:MyPipelineParams"


def main():
    """Main entrypoint that starts uvicorn with the AI Runner app."""
    # Set environment variables for the live pipeline
    os.environ.setdefault("PIPELINE", "live-video-to-video")
    os.environ.setdefault("MODEL_ID", "my-pipeline")
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
```

### Step 5: Install and Use

Install your packages:

```bash
# Development install
pip install -e /path/to/my-pipeline
pip install -e /path/to/my-pipeline-entrypoint

# Or from git repos
pip install git+https://github.com/yourorg/my-pipeline.git
pip install git+https://github.com/yourorg/my-pipeline-entrypoint.git
```

Then run:

```bash
my-pipeline-runner
```

## Docker Images

Create a Dockerfile that installs your entrypoint package:

```dockerfile
ARG BASE_IMAGE=livepeer/ai-runner:live-base
FROM ${BASE_IMAGE}

# Install Python via pyenv
ARG PYTHON_VERSION=3.10
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install ai-runner-base
COPY runner/pyproject.toml /app/pyproject.toml
RUN mkdir -p /app/app && touch /app/app/__init__.py
RUN pip install --no-cache-dir --no-build-isolation .

# Copy app code
COPY runner/app/ /app/app

# Re-install to register package
RUN pip install --no-cache-dir --no-build-isolation --no-deps .

# Install your pipeline and entrypoint
RUN pip install my-pipeline my-pipeline-entrypoint

# Runtime configuration
ENV PIPELINE="live-video-to-video" \
    MODEL_ID="my-pipeline"

CMD ["my-pipeline-runner"]
```

## Multi-Processing Considerations

Since pipelines run in a spawned subprocess:

1. **All dependencies must be installed** - The spawned process needs access to your pipeline package
2. **Import paths must be valid** - The import paths passed via CLI must be resolvable
3. **No shared state** - Each process has its own memory space
4. **Import paths** - Use absolute imports from `app.live.pipelines.interface`

## Testing Your Pipeline

Test locally:

```bash
# Install ai-runner-base in development mode
cd /path/to/ai-runner/runner
pip install -e .

# Install your pipeline and entrypoint
pip install -e /path/to/my-pipeline
pip install -e /path/to/my-pipeline-entrypoint

# Run your pipeline
my-pipeline-runner
```

## Troubleshooting

### Pipeline Not Found

- Check that your import path is correct (format: `module.path:ClassName`)
- Verify the package is installed: `pip list | grep my-pipeline`
- Test the import manually: `python -c "from my_pipeline.pipeline import MyPipeline"`

### Import Errors in Spawned Process

- Ensure all dependencies are installed
- Use absolute imports from `app.live.pipelines.interface`
- Check that `ai-runner-base` is installed

### Parameters Not Parsing

- Verify your import path for params is correct
- Check that your params class extends `BaseParams`
- If `PARAMS_IMPORT` is empty, `BaseParams` will be used (this is expected)

## Example: Complete Pipeline Package

See `pipelines/streamdiffusion-entrypoint/` in the ai-runner repository for a complete working example.

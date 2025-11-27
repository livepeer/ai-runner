# External Pipeline Development Guide

This guide explains how to create AI Runner pipelines that live in separate repositories and can be installed as dependencies.

## Overview

AI Runner supports external pipelines through Python entry points. This allows pipeline developers to:

1. **Maintain separate repositories** - Each pipeline can have its own repo
2. **Use ai-runner-base as a library** - Install `ai-runner-base` as a dependency
3. **Automatic discovery** - Pipelines are discovered via entry points, no code changes needed in ai-runner
4. **Multi-processing support** - Works seamlessly with the multiprocessing architecture

## Architecture

### Multi-Processing Context

AI Runner uses `multiprocessing` with `spawn` context for GPU memory isolation. This means:

- Pipelines run in a **separate spawned process**
- The pipeline code must be **importable** from installed packages
- Entry points are discovered **in the spawned process**, so they must be installed packages

### Entry Points

Pipelines register themselves via two entry point groups:

- **`ai_runner.pipeline`** - Pipeline class (loaded when pipeline is instantiated)
- **`ai_runner.pipeline_params`** - Params class (loaded for lightweight parameter parsing)

If no `ai_runner.pipeline_params` entry point exists, the loader falls back to `BaseParams`. This allows simple pipelines to skip custom params if they only need base parameters (width, height, show_reloading_frame).

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
└── tests/
```

### Step 2: Configure Package and Entry Points

In your `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-pipeline"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ai-runner-base>=0.1.0",
    # Your pipeline-specific dependencies
    "torch>=2.0.0",
    # ... other deps
]

[project.entry-points."ai_runner.pipeline"]
my-pipeline = "my_pipeline.pipeline:MyPipeline"

[project.entry-points."ai_runner.pipeline_params"]
my-pipeline = "my_pipeline.params:MyPipelineParams"
```

**Note**: The `ai_runner.pipeline_params` entry point is optional. If omitted, the loader will use `BaseParams`.

### Step 3: Implement Pipeline Interface

Create `src/my_pipeline/pipeline.py`:

```python
from app.live.pipelines.interface import Pipeline
from app.live.pipelines.trickle import VideoFrame, VideoOutput
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

### Step 4: Implement Parameters

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

### Step 5: Install and Use

Install your pipeline:

```bash
# Development install
pip install -e /path/to/my-pipeline

# Or from a git repo
pip install git+https://github.com/yourorg/my-pipeline.git

# Or publish to PyPI
pip install my-pipeline
```

Then use it:

```bash
PIPELINE=my-pipeline MODEL_ID=my-pipeline python -m app.main
```

## Docker Images

### Option 1: Install in Dockerfile

Create a Dockerfile that extends the ai-runner base image:

```dockerfile
FROM livepeer/ai-runner:live-base-streamdiffusion

# Install your pipeline
RUN pip install my-pipeline

# Or install from git
RUN pip install git+https://github.com/yourorg/my-pipeline.git
```

### Option 2: Install via uv (Recommended)

If using `uv` for faster installs:

```dockerfile
FROM livepeer/ai-runner:live-base-streamdiffusion

# Install uv if not already present
RUN pip install uv

# Install your pipeline
RUN uv pip install my-pipeline
```

**Note**: Choose the appropriate base image (`live-base-streamdiffusion`, `live-base-comfyui`, etc.) based on your pipeline's dependencies.

## Multi-Processing Considerations

Since pipelines run in a spawned subprocess:

1. **All dependencies must be installed** - The spawned process needs access to your pipeline package
2. **Entry points must be discoverable** - They're discovered in the spawned process
3. **No shared state** - Each process has its own memory space
4. **Import paths** - Use absolute imports from `app.live.pipelines.interface`

## Testing Your Pipeline

Test locally:

```bash
# Install ai-runner-base in development mode
cd /path/to/ai-runner/runner
pip install -e .

# Install your pipeline
cd /path/to/my-pipeline
pip install -e .

# Run the runner
cd /path/to/ai-runner/runner
PIPELINE=my-pipeline MODEL_ID=my-pipeline python -m app.main
```

## Publishing to PyPI

1. Build your package:
   ```bash
   python -m build
   ```

2. Publish:
   ```bash
   twine upload dist/*
   ```

3. Users can then install:
   ```bash
   pip install my-pipeline
   ```

## Advanced: Pipeline Variants

You can create pipeline variants (like `streamdiffusion-sd15`) by registering multiple entry points:

```toml
[project.entry-points."ai_runner.pipeline"]
my-pipeline = "my_pipeline.pipeline:MyPipeline"
my-pipeline-fast = "my_pipeline.pipeline:MyPipelineFast"
my-pipeline-hq = "my_pipeline.pipeline:MyPipelineHQ"

[project.entry-points."ai_runner.pipeline_params"]
my-pipeline = "my_pipeline.params:MyPipelineParams"
my-pipeline-fast = "my_pipeline.params:MyPipelineParams"
my-pipeline-hq = "my_pipeline.params:MyPipelineHQParams"
```

The loader will try exact match first, then fall back to base name (e.g., `my-pipeline-fast` → `my-pipeline`).

## Troubleshooting

### Pipeline Not Found

- Check that entry points are registered correctly in `pyproject.toml`
- Verify the package is installed: `pip list | grep my-pipeline`
- Check entry points: `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='ai_runner.pipeline')))"`

### Import Errors in Spawned Process

- Ensure all dependencies are installed
- Use absolute imports from `app.live.pipelines.interface`
- Check that `ai-runner-base` is installed

### Parameters Not Parsing

- Verify the params entry point is registered in `pyproject.toml`
- Check that your params class extends `BaseParams`
- Ensure params can be imported without expensive dependencies (torch, etc.)
- If params entry point is missing, the loader will use `BaseParams` (this is expected)

## Example: Complete Pipeline Package

See `examples/external-pipeline-example/` for a complete working example.

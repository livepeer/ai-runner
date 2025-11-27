# Pipeline Plugin Architecture

## Overview

This document describes the architecture for making AI Runner pipelines pluggable, allowing them to live in separate repositories and be installed as dependencies.

## Key Design Decisions

### 1. Entry Points for Discovery

We use Python entry points (`ai_runner.pipeline` and `ai_runner.pipeline_params`) to discover pipelines at runtime. This allows:

- **Zero code changes** in ai-runner when adding new pipelines
- **Dynamic discovery** - pipelines are found automatically when installed
- **Multi-process compatibility** - entry points work in spawned subprocesses

### 2. Multi-Processing Compatibility

Since pipelines run in spawned subprocesses (`mp.get_context("spawn")`):

- Pipelines must be **installed packages** (not just Python files)
- Entry points are discovered **in the spawned process**
- All dependencies must be installed and importable

### 3. Package Structure

The `ai-runner-base` package provides:

- `app.live.pipelines.interface.Pipeline` - Abstract base class
- `app.live.pipelines.interface.BaseParams` - Base parameter class
- `app.live.pipelines.trickle` - Frame types (VideoFrame, VideoOutput, etc.)
- Runtime infrastructure (process management, queues, etc.)

External pipelines depend on `ai-runner-base` and implement the `Pipeline` interface.

## How It Works

### Pipeline Discovery Flow

```
1. Runner starts with PIPELINE=my-pipeline
2. PipelineProcess.start("my-pipeline") is called
3. In spawned process, loader.py discovers entry points:
   - Checks installed packages for "ai_runner.pipeline" entry points
   - Finds "my-pipeline" entry point
   - Loads the pipeline class
4. Pipeline is instantiated and initialized
```

### Entry Point Registration

Pipelines register themselves in `pyproject.toml`:

```toml
[project.entry-points."ai_runner.pipeline"]
my-pipeline = "my_pipeline.pipeline:MyPipeline"

[project.entry-points."ai_runner.pipeline_params"]
my-pipeline = "my_pipeline.params:MyPipelineParams"
```

The params entry point points directly to the params class:
- A factory function: `def create_params(params: dict) -> BaseParams`
- A Pydantic class: The loader will instantiate it with `**params`

## Docker Image Strategy

### Current Approach

Currently, each pipeline has its own Docker image:
- `livepeer/ai-runner:live-base-streamdiffusion`
- `livepeer/ai-runner:live-app-streamdiffusion`
- `livepeer/ai-runner:live-base-scope`
- `livepeer/ai-runner:live-app-scope`

### Proposed Approaches

#### Option 1: Single Base Image + Runtime Install (Recommended)

Use a single base image and install pipelines at runtime:

```dockerfile
FROM livepeer/ai-runner:live-base

# Install pipeline via uv (fast)
RUN uv pip install my-pipeline

# Or install from git
RUN uv pip install git+https://github.com/org/my-pipeline.git
```

**Pros:**
- Single base image to maintain
- Pipelines can be updated without rebuilding images
- Works with PyPI packages

**Cons:**
- Requires network access at build time
- Slower container startup if installing at runtime

#### Option 2: Pipeline-Specific Base Images

Each pipeline provides its own base image with dependencies:

```dockerfile
# In my-pipeline repo
FROM livepeer/ai-runner:live-base

# Install pipeline-specific system dependencies
RUN apt-get install -y some-system-lib

# Install Python dependencies
RUN uv pip install my-pipeline

# ai-runner code is copied in final stage
FROM my-pipeline-base
COPY --from=ai-runner /app /app
```

**Pros:**
- Pipelines control their own dependencies
- Can optimize images per pipeline

**Cons:**
- More images to maintain
- Harder to update ai-runner base

#### Option 3: Pure Python Dependencies (Best Case)

If pipelines only need Python dependencies:

```dockerfile
FROM livepeer/ai-runner:live-base

# Set environment variable for auto-install
ENV AUTO_INSTALL_PIPELINE="my-pipeline"

# Startup script installs pipeline if needed
COPY install-pipeline.sh /usr/local/bin/
```

Then modify runner startup to check `AUTO_INSTALL_PIPELINE` and install automatically.

**Pros:**
- No Docker rebuilds needed
- Pipelines can be installed from PyPI
- Works with `uv` for fast installs

**Cons:**
- Only works for pure Python dependencies
- Requires network at runtime

## Implementation Status

✅ **Completed:**
- Entry point discovery system
- Pipeline loader refactoring
- Base package structure (`pyproject.toml`)
- Documentation

🔄 **In Progress:**
- Factory functions for built-in pipeline params
- Example external pipeline package
- Docker image strategy documentation

📋 **Future Work:**
- Auto-install pipeline support (`AUTO_INSTALL_PIPELINE` env var)
- Pipeline dependency resolution
- Pipeline registry/metadata system
- Testing framework for external pipelines

## Migration Path

### For Existing Pipelines (streamdiffusion, scope, etc.)

1. **Keep current Docker images** - No breaking changes
2. **Add entry points** - Register in `pyproject.toml` (already done)
3. **Gradually migrate** - Can move to separate repos over time

### For New Pipelines

1. **Create separate repo** - `my-pipeline/`
2. **Depend on ai-runner-base** - `dependencies = ["ai-runner-base>=0.1.0"]`
3. **Register entry points** - In `pyproject.toml`
4. **Install and use** - `pip install my-pipeline` then `PIPELINE=my-pipeline`

## Example: External Pipeline Package

See `docs/external-pipelines.md` for a complete example.

## Questions & Answers

### Q: How do pipelines link to their implementation given multiprocessing?

**A:** Entry points are discovered in the spawned process. Since we use `spawn` context, the process starts fresh and imports packages. As long as the pipeline package is installed, its entry points will be discoverable.

### Q: How do pipelines provide their own Docker images?

**A:** Three options:
1. **Runtime install** - Install pipeline in Dockerfile or at startup
2. **Pipeline base image** - Pipeline provides base image with deps, ai-runner adds its code
3. **Pure Python** - Use `AUTO_INSTALL_PIPELINE` env var for automatic install

### Q: Can we avoid Docker entirely?

**A:** For pure Python dependencies, yes! Use `uv pip install my-pipeline` at runtime. For system dependencies, you'll still need Docker, but pipelines can specify their own base images.

### Q: How do we handle pipeline dependencies?

**A:** Pipelines declare dependencies in their `pyproject.toml`. When installed via `pip` or `uv`, dependencies are resolved automatically. For Docker, pipelines can either:
- Install in their Dockerfile
- Use a base image with dependencies pre-installed
- Rely on runtime install (if pure Python)


# Pipeline Loading Architecture

## Overview

This document describes the architecture for making AI Runner pipelines pluggable, allowing them to live in separate repositories and be installed as dependencies.

## Key Design Decisions

### 1. Import-Based Loading

We use explicit import paths passed via CLI arguments to load pipelines at runtime:

- `--pipeline-import` - Full import path for Pipeline class (e.g., `app.live.pipelines.noop:Noop`)
- `--params-import` - Full import path for Params class (optional - empty uses `BaseParams`)

This allows:

- **Zero code changes** in ai-runner when adding new pipelines
- **Simple configuration** - just pass import paths via env vars or CLI
- **Multi-process compatibility** - import paths work in spawned subprocesses

### 2. Multi-Processing Compatibility

Since pipelines run in spawned subprocesses (`mp.get_context("spawn")`):

- Pipelines must be **installed packages** (not just Python files)
- Import paths are passed via CLI arguments to the spawned process
- All dependencies must be installed and importable

### 3. Package Structure

The `ai-runner-base` package provides:

- `app.live.pipelines.interface.Pipeline` - Abstract base class
- `app.live.pipelines.interface.BaseParams` - Base parameter class
- `app.live.trickle` - Frame types (VideoFrame, VideoOutput, etc.)
- Runtime infrastructure (process management, queues, etc.)

External pipelines depend on `ai-runner-base` and implement the `Pipeline` interface.

### 4. Entrypoint Packages

Each pipeline has a separate entrypoint package that:

- Depends on `ai-runner-base` and the pipeline library
- Sets environment variables (`PIPELINE_IMPORT`, `PARAMS_IMPORT`)
- Provides a CLI command to start the runner

This keeps the pipeline code separate from the startup configuration.

## How It Works

### Pipeline Loading Flow

```
1. Entrypoint script sets PIPELINE_IMPORT and PARAMS_IMPORT env vars
2. Entrypoint calls uvicorn with app.main:app
3. main.py creates LiveVideoToVideoPipeline with env var values
4. LiveVideoToVideoPipeline starts infer.py subprocess with --pipeline-import and --params-import
5. infer.py passes import paths to ProcessGuardian
6. In PipelineProcess (spawned subprocess):
   - loader.py imports the Pipeline class using the import path
   - loader.py imports the Params class (or uses BaseParams if empty)
   - Pipeline is instantiated and initialized
```

### Import Path Format

Import paths use the format `module.path:ClassName`:

```
app.live.pipelines.noop:Noop
app.live.pipelines.streamdiffusion.pipeline:StreamDiffusion
my_pipeline.pipeline:MyPipeline
```

The loader splits on `:` and uses `importlib.import_module` to load the class.

## Docker Image Strategy

### Current Approach

Each pipeline has its own final Docker image:

```
Dockerfile.live-base                    # Common base: CUDA, pyenv, FFmpeg
    ├── Dockerfile.live-app-streamdiffusion  # + PyTorch, StreamDiffusion, entrypoint
    ├── Dockerfile.live-app-comfyui          # + ComfyUI, entrypoint
    ├── Dockerfile.live-app-scope            # + Scope, entrypoint
    └── Dockerfile.live-app-noop             # + CPU PyTorch, built-in noop
```

Each final image:
1. Installs pipeline-specific libraries (PyTorch, StreamDiffusion, etc.)
2. Installs `ai-runner-base` package
3. Installs the pipeline's entrypoint package
4. Sets CMD to the pipeline's CLI command (e.g., `streamdiffusion-runner`)

### External Pipeline Images

External pipelines can create their own Docker images:

```dockerfile
FROM livepeer/ai-runner:live-base

# Install Python and PyTorch
RUN pyenv install 3.10 && pyenv global 3.10
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install ai-runner-base
COPY runner/pyproject.toml /app/pyproject.toml
RUN pip install --no-build-isolation .
COPY runner/app/ /app/app
RUN pip install --no-build-isolation --no-deps .

# Install your pipeline and entrypoint
RUN pip install my-pipeline my-pipeline-entrypoint

CMD ["my-pipeline-runner"]
```

## Implementation Status

✅ **Completed:**
- Import-based pipeline loading (`--pipeline-import`, `--params-import`)
- Entrypoint packages for each pipeline
- Simplified loader (no entry point discovery)
- Docker image restructuring
- Documentation

📋 **Future Work:**
- Auto-install pipeline support (`AUTO_INSTALL_PIPELINE` env var)
- Pipeline dependency resolution
- Pipeline registry/metadata system
- Testing framework for external pipelines

## Migration Path

### For Existing Pipelines (streamdiffusion, scope, etc.)

Existing pipelines have been migrated to the new structure:
1. Pipeline code remains in `runner/app/live/pipelines/`
2. Entrypoint packages created in `pipelines/{name}-entrypoint/`
3. Docker images renamed to `Dockerfile.live-app-{pipeline}`

### For New Pipelines

1. **Create separate repo** - `my-pipeline/`
2. **Depend on ai-runner-base** - `dependencies = ["ai-runner-base>=0.1.0"]`
3. **Create entrypoint package** - Sets import paths and starts uvicorn
4. **Create Dockerfile** - Installs your packages
5. **Run** - `my-pipeline-runner`

## Example: External Pipeline Package

See `docs/external-pipelines.md` for a complete example.

## Questions & Answers

### Q: How do pipelines link to their implementation given multiprocessing?

**A:** Import paths are passed as CLI arguments to the spawned process. Since we use `spawn` context, the process starts fresh and imports packages. As long as the pipeline package is installed, it will be importable.

### Q: How do pipelines provide their own Docker images?

**A:** Pipelines create a Dockerfile that:
1. Starts FROM `livepeer/ai-runner:live-base`
2. Installs their dependencies
3. Installs `ai-runner-base`
4. Installs their entrypoint package
5. Sets CMD to their runner command

### Q: Can we avoid Docker entirely?

**A:** For development, yes! Install `ai-runner-base` and your pipeline in a virtual environment, then run your entrypoint command. For production, Docker is recommended for reproducibility.

### Q: How do we handle pipeline dependencies?

**A:** Pipelines declare dependencies in their `pyproject.toml`. When installed via `pip` or `uv`, dependencies are resolved automatically. The entrypoint package should depend on both `ai-runner-base` and the pipeline library.

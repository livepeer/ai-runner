# Pipeline Separation Summary

## Problem Statement

You want to make it so that developers of specific pipelines (like `scope` and `streamdiffusion`) have to interact the least possible with the outer runtime. Ideally:
- Pipelines could live in separate repositories
- They only use runner base as a lib/app wrapper
- They can provide their own Docker images (or ideally avoid Docker entirely)
- Works with the multi-processing approach

## Solution Implemented

### ✅ Import-Based Pipeline Loading

**What Changed:**
- Refactored `loader.py` to load pipelines via explicit import paths
- Added `--pipeline-import` and `--params-import` CLI flags to `infer.py`
- Created entrypoint packages that set environment variables and start the runner
- No entry point registration needed - just pass import paths

**How It Works:**
1. Entrypoint package sets `PIPELINE_IMPORT` and `PARAMS_IMPORT` environment variables
2. `main.py` passes these to `LiveVideoToVideoPipeline`
3. `infer.py` receives import paths via CLI and passes to `ProcessGuardian`
4. `PipelineProcess` loads the pipeline class using `importlib.import_module`

### ✅ Multi-Processing Compatibility

**Question:** How do pipelines link to their implementation given multiprocessing?

**Answer:** Import paths are passed as CLI arguments to the spawned `infer.py` process. Since we use `mp.get_context("spawn")`, the process starts fresh and imports installed packages. As long as the pipeline package is installed (via `pip` or `uv`), it will be importable.

**Key Point:** Pipelines must be **installed packages**, not just Python files.

### ✅ Docker Image Strategy

**Question:** How do pipelines provide their own Docker images?

**Current Implementation:**

Each pipeline has its own final Docker image:
```
Dockerfile.live-base                    # Common base: CUDA, pyenv, FFmpeg
    ├── Dockerfile.live-app-streamdiffusion  # + PyTorch, StreamDiffusion, entrypoint
    ├── Dockerfile.live-app-comfyui          # + ComfyUI, entrypoint
    ├── Dockerfile.live-app-scope            # + Scope, entrypoint
    └── Dockerfile.live-app-noop             # + CPU PyTorch, built-in noop
```

Each final image:
1. Starts FROM `livepeer/ai-runner:live-base`
2. Installs Python and pipeline-specific libraries
3. Installs `ai-runner-base` package
4. Installs the entrypoint package
5. Sets CMD to the entrypoint command (e.g., `streamdiffusion-runner`)

**For External Pipelines:**

```dockerfile
FROM livepeer/ai-runner:live-base

# Install Python and dependencies
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

### ✅ Avoiding Docker Entirely (Development)

For pure Python dependencies:

```bash
# Install ai-runner-base
cd ai-runner/runner
pip install -e .

# Install your pipeline and entrypoint
pip install -e /path/to/my-pipeline
pip install -e /path/to/my-pipeline-entrypoint

# Run
my-pipeline-runner
```

**Limitation:** Only works if pipeline has no system dependencies. For production, Docker is recommended.

## Migration Path

### For Existing Pipelines (scope, streamdiffusion)

Existing pipelines have been migrated:
1. Pipeline code remains in `runner/app/live/pipelines/`
2. Entrypoint packages created in `pipelines/{name}-entrypoint/`
3. Docker images renamed to `Dockerfile.live-app-{pipeline}`

### For New Pipelines

1. Create separate repo: `my-pipeline/`
2. Depend on `ai-runner-base`: `dependencies = ["ai-runner-base>=0.1.0"]`
3. Implement `Pipeline` interface
4. Create entrypoint package with import paths
5. Create Dockerfile
6. Run: `my-pipeline-runner`

## Example: External Pipeline Structure

```
my-pipeline/
├── pyproject.toml
├── src/
│   └── my_pipeline/
│       ├── __init__.py
│       ├── pipeline.py     # Implements Pipeline interface
│       └── params.py       # Extends BaseParams
└── my-pipeline-entrypoint/
    ├── pyproject.toml
    └── src/
        └── my_pipeline_entrypoint/
            └── __init__.py  # Sets PIPELINE_IMPORT/PARAMS_IMPORT, starts uvicorn
```

**Key imports:**
```python
from app.live.pipelines.interface import Pipeline, BaseParams
from app.live.trickle import VideoFrame, VideoOutput
```

## Files Changed

- ✅ `runner/app/live/infer.py` - Added `--pipeline-import` and `--params-import` flags
- ✅ `runner/app/live/pipelines/loader.py` - Import-based loading (no entry points)
- ✅ `runner/app/pipelines/live_video_to_video.py` - Passes import paths to subprocess
- ✅ `runner/app/main.py` - Reads `PIPELINE_IMPORT`/`PARAMS_IMPORT` env vars
- ✅ `runner/pyproject.toml` - Removed entry point declarations
- ✅ `pipelines/*/` - Entrypoint packages for each pipeline
- ✅ `runner/docker/Dockerfile.live-app-*` - Final images with entrypoints
- ✅ `docs/external-pipelines.md` - Complete guide for external pipelines
- ✅ `docs/pipeline-plugin-architecture.md` - Architecture overview

## Benefits

1. **Separation of concerns** - Pipelines are independent packages
2. **Simple configuration** - Just set import paths via env vars
3. **Multi-process compatible** - Works with spawned subprocesses
4. **Flexible deployment** - Can use Docker, PyPI, or git installs
5. **Easy testing** - Pipelines can be developed and tested independently

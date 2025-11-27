# Pipeline Separation Summary

## Problem Statement

You want to make it so that developers of specific pipelines (like `scope` and `streamdiffusion`) have to interact the least possible with the outer runtime. Ideally:
- Pipelines could live in separate repositories
- They only use runner base as a lib/app wrapper
- They can provide their own Docker images (or ideally avoid Docker entirely)
- Works with the multi-processing approach

## Solution Implemented

### ✅ Entry Point-Based Plugin System

**What Changed:**
- Refactored `loader.py` to discover pipelines via Python entry points
- Created `pyproject.toml` for `ai-runner-base` package
- Pipelines register themselves via entry points, no code changes needed in ai-runner

**How It Works:**
1. Pipelines register entry points in their `pyproject.toml`:
   ```toml
   [project.entry-points."ai_runner.pipeline"]
   my-pipeline = "my_pipeline.pipeline:MyPipeline"
   ```
2. Runner discovers pipelines automatically when installed
3. Works seamlessly with multiprocessing (entry points discovered in spawned process)

### ✅ Multi-Processing Compatibility

**Question:** How do pipelines link to their implementation given multiprocessing?

**Answer:** Entry points are discovered **in the spawned process**. Since we use `mp.get_context("spawn")`, the process starts fresh and imports installed packages. As long as the pipeline package is installed (via `pip` or `uv`), its entry points will be discoverable.

**Key Point:** Pipelines must be **installed packages**, not just Python files. This is already the case with the entry point system.

### ✅ Docker Image Strategy

**Question:** How do pipelines provide their own Docker images?

**Three Options:**

#### Option 1: Runtime Install (Recommended for Pure Python)
```dockerfile
FROM livepeer/ai-runner:live-base
RUN uv pip install my-pipeline
```
- Single base image
- Pipelines installed at build time
- Works with PyPI packages

#### Option 2: Pipeline Base Image
```dockerfile
# In my-pipeline repo
FROM livepeer/ai-runner:live-base
RUN apt-get install -y pipeline-deps
RUN uv pip install my-pipeline
```
- Pipeline controls its own dependencies
- ai-runner code added in final stage
- More images to maintain

#### Option 3: Auto-Install at Runtime (Best Case)
```dockerfile
FROM livepeer/ai-runner:live-base
ENV AUTO_INSTALL_PIPELINE="my-pipeline"
```
Then modify runner to check `AUTO_INSTALL_PIPELINE` and install automatically.

**For your use case:** Option 1 or 3 work best. Option 3 is ideal if pipelines are pure Python dependencies.

### ✅ Avoiding Docker Entirely

**Question:** Can we avoid Docker and use `uv` dependency installed automatically?

**Answer:** Yes! For pure Python dependencies:

1. **Install pipeline via uv:**
   ```bash
   uv pip install my-pipeline
   ```

2. **Or use environment variable:**
   ```bash
   AUTO_INSTALL_PIPELINE="my-pipeline" python -m app.main
   ```
   (Requires implementing auto-install logic in runner startup)

3. **Or install in Dockerfile:**
   ```dockerfile
   FROM livepeer/ai-runner:live-base
   RUN uv pip install my-pipeline
   ```

**Limitation:** Only works if pipeline has no system dependencies. For system deps, you'll still need Docker, but pipelines can specify their own base images.

## Migration Path

### For Existing Pipelines (scope, streamdiffusion)

1. **No breaking changes** - Current Docker images still work
2. **Entry points already registered** - In `pyproject.toml`
3. **Can migrate gradually** - Move to separate repos when ready

### For New Pipelines

1. Create separate repo: `my-pipeline/`
2. Depend on `ai-runner-base`: `dependencies = ["ai-runner-base>=0.1.0"]`
3. Implement `Pipeline` interface
4. Register entry points in `pyproject.toml`
5. Install: `pip install my-pipeline` or `uv pip install my-pipeline`
6. Use: `PIPELINE=my-pipeline MODEL_ID=my-pipeline python -m app.main`

## Example: External Pipeline Structure

```
my-pipeline/
├── pyproject.toml          # Entry points registered here
├── src/
│   └── my_pipeline/
│       ├── __init__.py
│       ├── pipeline.py     # Implements Pipeline interface
│       └── params.py        # Extends BaseParams
└── README.md
```

**Key imports:**
```python
from app.live.pipelines.interface import Pipeline, BaseParams
from app.live.pipelines.trickle import VideoFrame, VideoOutput
```

## Next Steps

1. **Test the entry point system** - Verify existing pipelines still work
2. **Create example external pipeline** - Show how to create a new pipeline in a separate repo
3. **Implement auto-install** (optional) - Add `AUTO_INSTALL_PIPELINE` support
4. **Update Dockerfiles** - Show how to install external pipelines
5. **Documentation** - Update main README with plugin system info

## Files Changed

- ✅ `runner/pyproject.toml` - Package definition with entry points
- ✅ `runner/app/live/pipelines/loader.py` - Entry point discovery
- ✅ `docs/external-pipelines.md` - Complete guide for external pipelines
- ✅ `docs/pipeline-plugin-architecture.md` - Architecture overview

## Benefits

1. **Separation of concerns** - Pipelines are independent packages
2. **No code changes in ai-runner** - New pipelines just need to be installed
3. **Multi-process compatible** - Works with spawned subprocesses
4. **Flexible deployment** - Can use Docker, PyPI, or git installs
5. **Easy testing** - Pipelines can be developed and tested independently


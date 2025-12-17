# AGENTS.md - AI Runner Project Context

> **Purpose**: High-level context for AI agents. Read this first to understand the project's architecture, patterns, and conventions before making changes.
>
> **Related Documents**:
> - [`README.md`](./README.md) - Project overview and basic usage
> - **AGENTS.md** (this file) - Current implementation state, patterns, and conventions
> - [`docs/`](./docs/) - Detailed implementation guides and development docs

## 🚀 TL;DR - Start Here

**What is this?** The AI Runner is a containerized Python application that processes AI inference jobs on the Livepeer network. This document focuses on the **live video-to-video pipeline**.

**Tech Stack**: Python + FastAPI + PyTorch + Multiprocessing + Trickle protocol

**Core Flow**: Video Stream → Trickle Subscriber → FFmpeg Decode → Pipeline Process → AI Inference → FFmpeg Encode → Trickle Publisher → Video Stream

**Key Files to Know**:
- `runner/src/runner/main.py` - FastAPI entrypoint, loads pipelines
- `runner/src/runner/pipelines/live_video_to_video.py` - Starts `infer.py` subprocess
- `runner/src/runner/live/infer.py` - Main inference process orchestrator
- `runner/src/runner/live/process/process.py` - Isolated multiprocessing pipeline execution
- `runner/src/runner/live/pipelines/` - Actual AI pipeline implementations

**Need Details?** Check the Quick Navigation Guide below or `docs/` folder

---

## 📖 How to Use This Document

**AGENTS Philosophy**:
- **Stay Concise**: Focus on high-level overview, not implementation details
- **Preserve Intent**: Document historical decisions and quirks so they're not forgotten
- **Point to Details**: Reference specific files and functions for deep-dives
- **Map the Territory**: Clearly outline project structure so agents know where to look

**When making changes**:
1. Read this file first for overall context
2. Check `docs/` folder for detailed guides on specific features
3. Update AGENTS.md only for architectural changes, not implementation details
4. Use `grep` and `codebase_search` to understand code before modifying

**⚠️ CRITICAL FOR AI AGENTS**: When you complete significant changes (new features, architectural modifications, workflow updates), you MUST update AGENTS.md to reflect those changes. This file is the primary context source for future agents.

### Quick Navigation Guide

| **Looking for...** | **Go to...** |
|-------------------|-------------|
| Overall architecture | This file (AGENTS.md) |
| Runtime overview | `docs/live-ai-runtime-overview.md` |
| Local development | `docs/live-ai-local-dev.md` |
| Container setup | `docs/runner-docker.md` |
| FastAPI entrypoint | `runner/src/runner/main.py` |
| Live pipeline wrapper | `runner/src/runner/pipelines/live_video_to_video.py` |
| Infer process | `runner/src/runner/live/infer.py` |
| Process management | `runner/src/runner/live/process/` |
| Streaming protocols | `runner/src/runner/live/streamer/` |
| Frame encoding/decoding | `runner/src/runner/live/trickle/` |
| Pipeline implementations | `runner/src/runner/live/pipelines/` |
| HTTP API (internal) | `runner/src/runner/live/api/api.py` |

---

## 🎯 Project Mission

**AI Runner** is the containerized inference runtime for the Livepeer AI network. For live video-to-video processing, it receives video streams, applies real-time AI transformations (e.g., style transfer, diffusion effects), and outputs the transformed stream with minimal latency.

---

## 🏗️ Tech Stack

- **Runtime**: Python 3.11+, FastAPI (container API), aiohttp (internal API)
- **AI/ML**: PyTorch, CUDA, StreamDiffusion, ComfyUI
- **Concurrency**: asyncio, Python multiprocessing (spawn mode)
- **Streaming**: Trickle protocol (low-latency HTTP-based streaming)
- **Media**: FFmpeg (via PyAV), tensor-based frame processing
- **Container**: Docker with NVIDIA GPU support

---

## 📐 Architecture Overview

### Two-Level Process Hierarchy

```
Container Start
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  main.py (FastAPI)                                          │
│  - Container entrypoint                                     │
│  - Loads pipeline based on PIPELINE env var                 │
│  - Exposes /live-video-to-video, /health, /metrics          │
└─────────────────────────────────────────────────────────────┘
    │
    │ (subprocess via subprocess.Popen)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  infer.py (aiohttp)                                         │
│  - ProcessGuardian: monitors pipeline health                │
│  - PipelineStreamer: handles Trickle ingress/egress         │
│  - Internal HTTP API on port 8888                           │
└─────────────────────────────────────────────────────────────┘
    │
    │ (multiprocessing via mp.Process)
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PipelineProcess (isolated subprocess)                      │
│  - Loads actual AI pipeline (StreamDiffusion, ComfyUI, etc) │
│  - Processes frames in tight loop                           │
│  - Communicates via multiprocessing.Queue                   │
└─────────────────────────────────────────────────────────────┘
```

### Why Two-Level Isolation?

1. **GPU Memory Isolation**: If the pipeline crashes (OOM, CUDA errors), only the PipelineProcess dies. The infer.py process can restart it without killing the container.

2. **Restart Capability**: infer.py can restart PipelineProcess up to 3 times before giving up. This handles transient errors gracefully.

3. **Clean Shutdown**: Parent death signals and watchdogs ensure child processes terminate when parents die.

4. **Performance**: The innermost process (PipelineProcess) uses minimal IPC (multiprocessing.Queue) so inference isn't blocked by network I/O.

### Data Flow

Streams are started via `POST /api/live-video-to-video` which creates ad-hoc Trickle connections:

```
POST /api/live-video-to-video
    │ (creates TrickleProtocol with subscribe_url, publish_url)
    ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ TrickleSubscriber│ ──▶ │  FFmpeg Decode  │ ──▶ │  Input Queue    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Pipeline Process │
                                                │  (AI Inference)  │
                                                └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ TricklePublisher│ ◀── │  FFmpeg Encode  │ ◀── │  Output Queue   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
    │
    ▼
External Trickle Stream (to Orchestrator)
```

---

## 🗂️ Project Structure

### Container Entrypoint (`runner/src/runner/main.py`)

```python
# Key responsibilities:
# 1. Load pipeline based on PIPELINE env var
# 2. Start FastAPI server with routes
# 3. Expose /health, /metrics endpoints

def load_pipeline(pipeline: str, model_id: str):
    match pipeline:
        case "live-video-to-video":
            from runner.pipelines.live_video_to_video import LiveVideoToVideoPipeline
            return LiveVideoToVideoPipeline(model_id)
        # ... other pipelines
```

### Live Video Pipeline Wrapper (`runner/src/runner/pipelines/live_video_to_video.py`)

```python
class LiveVideoToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        # Starts infer.py as subprocess
        self.start_process()

    def __call__(self, *, subscribe_url, publish_url, control_url, events_url, params, ...):
        # Forwards stream request to infer.py via HTTP
        conn.request("POST", "/api/live-video-to-video", ...)

    def get_health(self):
        # Proxies health check to infer.py
        conn.request("GET", "/api/status")
```

### Infer Process (`runner/src/runner/live/infer.py`)

```python
async def main(...):
    # 1. Start ProcessGuardian (manages PipelineProcess)
    process = ProcessGuardian(pipeline, params)
    await process.start()

    # 2. Start PipelineStreamer (handles Trickle I/O)
    streamer = PipelineStreamer(protocol, process, ...)
    await streamer.start(params)

    # 3. Start internal HTTP API
    api = await start_http_server(http_port, process, streamer)

    # 4. Wait for shutdown signal
    await asyncio.wait([signal_task, exception_task, streamer.wait()], ...)
```

### Process Management (`runner/src/runner/live/process/`)

| File | Purpose |
|------|---------|
| `process_guardian.py` | Monitors pipeline health, handles restarts, computes state |
| `process.py` | `PipelineProcess` class - multiprocessing wrapper for pipelines |
| `status.py` | State machine: `LOADING`, `ONLINE`, `DEGRADED_*`, `ERROR`, `OFFLINE` |
| `loading_overlay.py` | Renders "loading" frame when pipeline is reinitializing |

### Streaming (`runner/src/runner/live/streamer/`)

| File | Purpose |
|------|---------|
| `streamer.py` | `PipelineStreamer` - orchestrates ingress/egress/control loops |
| `protocol/protocol.py` | Abstract `StreamProtocol` interface |
| `protocol/trickle.py` | Trickle protocol implementation (production) |
| `protocol/zeromq.py` | ⚠️ **DEPRECATED** - Was used for local development only, no longer maintained |

### Frame Handling (`runner/src/runner/live/trickle/`)

| File | Purpose |
|------|---------|
| `frame.py` | `VideoFrame`, `AudioFrame`, `VideoOutput`, `AudioOutput` classes |
| `media.py` | FFmpeg subprocess management for encode/decode |
| `encoder.py` | Frame → FFmpeg → Trickle |
| `decoder.py` | Trickle → FFmpeg → Frame |
| `trickle_publisher.py` | Trickle HTTP client for publishing |
| `trickle_subscriber.py` | Trickle HTTP client for subscribing |

### Pipeline Implementations (`runner/src/runner/live/pipelines/`)

| Pipeline | Files | Description |
|----------|-------|-------------|
| `streamdiffusion` | `streamdiffusion/` | Real-time diffusion via StreamDiffusion library |
| `comfyui` | `comfyui/` | ComfyUI workflow execution |
| `scope` | `scope/` | Scope pipeline |
| `noop` | `noop.py` | Pass-through (for testing) |

**Pipeline Interface** (`interface.py`):
```python
class Pipeline(ABC):
    async def initialize(self, **params): ...
    async def put_video_frame(self, frame: VideoFrame, request_id: str): ...
    async def get_processed_video_frame(self) -> VideoOutput: ...
    async def update_params(self, **params) -> Task[None] | None: ...
    async def stop(self): ...
    @classmethod
    def prepare_models(cls): ...
```

---

## 🔑 Key Concepts

### Hard Constraints (Non-Negotiable)

- ✅ **Two-level process isolation** - PipelineProcess runs in spawned subprocess for GPU memory isolation
- ✅ **No blocking in infer.py** - All I/O must be async; blocking operations must use `asyncio.to_thread`
- ✅ **Queue-based IPC** - PipelineProcess communicates via `multiprocessing.Queue` only
- ✅ **Trickle protocol only** - All streaming uses Trickle (ad-hoc connections per stream)
- ✅ **Health state machine** - Must report accurate state for worker container management
- ✅ **Tensor format consistency** - Input: `(B, H, W, C)` range `[-1, 1]`, pipelines may convert internally

### State Machine

```
                        ┌─────────────────┐
                        │     LOADING     │ ◀─── Pipeline initializing
                        └────────┬────────┘
                                 │ ready
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │                        ONLINE                             │
    │                 (healthy, processing)                     │
    └─────┬──────────────────┬───────────────────┬─────────────┘
          │                  │                   │
          │ no input         │ inference slow    │ crash/timeout
          ▼                  ▼                   ▼
    ┌───────────┐     ┌───────────────┐    ┌───────────┐
    │ DEGRADED  │     │   DEGRADED    │    │   ERROR   │
    │   INPUT   │     │  INFERENCE    │    │           │
    └─────┬─────┘     └───────────────┘    └─────┬─────┘
          │                                      │
          │ 60s idle                             │ restart or kill
          ▼                                      ▼
    ┌───────────┐                          Container restart
    │  OFFLINE  │
    └───────────┘
```

### Frame Types

```python
# Input frames (from decoder)
class VideoFrame:
    tensor: torch.Tensor  # (B, H, W, C), float32, range [-1, 1]
    timestamp: int        # PTS in time_base units
    time_base: Fraction   # e.g., Fraction(1, 90000)
    log_timestamps: dict  # Performance tracking

class AudioFrame:
    samples: np.ndarray   # Audio samples
    format: str           # e.g., "fltp"
    rate: int             # Sample rate
    layout: str           # e.g., "stereo"

# Output frames (to encoder)
class VideoOutput:
    frame: VideoFrame
    request_id: str       # For request correlation
    is_loading_frame: bool

class AudioOutput:
    frames: List[AudioFrame]
    request_id: str
```

### Parameter Updates

Pipelines support dynamic parameter updates without restart:

```python
async def update_params(self, **params) -> Task[None] | None:
    """
    Update pipeline parameters.

    Returns:
        None if update was immediate (no loading overlay needed)
        Task if update requires pipeline reload (loading overlay shown)
    """
```

The ProcessGuardian handles the loading overlay when a Task is returned.

---

## 🎨 Code Patterns

### Error Handling

```python
# In PipelineProcess - errors are reported to error_queue
def _report_error(self, msg: str, error: Exception | None = None):
    error_event = {"message": f"{msg}: {error}", "timestamp": time.time()}
    self._try_queue_put(self.error_queue, error_event)

# In ProcessGuardian - errors trigger state changes and potential restarts
if state == PipelineState.ERROR:
    if restart_count >= 3:
        raise Exception("Pipeline process max restarts reached")
    await self._restart_process()
```

### Queue Operations (Non-Blocking)

```python
# Always use non-blocking puts with fallback
def _try_queue_put(self, _queue: mp.Queue, item: Any):
    try:
        _queue.put_nowait(item)
    except queue.Full:
        pass  # Drop item rather than block

# Use timeouts for blocking gets
async def recv_output(self) -> OutputFrame | None:
    while not self.is_done():
        try:
            return await asyncio.to_thread(self.output_queue.get, timeout=0.1)
        except queue.Empty:
            continue
```

### Graceful Shutdown

```python
# Signal handling in child process
def _setup_signal_handlers(done: mp.Event):
    def _handle(sig, _frame):
        done.set()
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

# Parent death watchdog
def _start_parent_watchdog(done: mp.Event):
    def _watch_parent():
        while not done.is_set():
            time.sleep(1)
            if os.getppid() == 1:  # Parent died (init is now parent)
                done.set()
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PIPELINE` | Pipeline type to load | `live-video-to-video` |
| `MODEL_ID` | Model/pipeline variant | `streamdiffusion`, `comfyui` |
| `INFERPY_INITIAL_PARAMS` | Initial pipeline params (JSON) | `{"prompt": "cyberpunk style"}` |
| `VERBOSE_LOGGING` | Enable debug logs | `1` |
| `HUGGINGFACE_HUB_CACHE` | Model cache directory | `/models` |
| `COMFY_UI_WORKSPACE` | ComfyUI installation path | `/comfyui` |

### Internal HTTP API (port 8888)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/live-video-to-video` | POST | Start a new stream (creates ad-hoc Trickle connections) |
| `/api/params` | POST | Update pipeline parameters |
| `/api/status` | GET | Get pipeline status |

### Stream Start Flow

The only way to start a stream is via the internal API. When `POST /api/live-video-to-video` is called:

1. **Receives stream URLs** - `subscribe_url`, `publish_url`, `control_url`, `events_url`
2. **Creates TrickleProtocol** - Ad-hoc Trickle connections are established for this stream
3. **Starts PipelineStreamer** - Orchestrates ingress/egress/control loops
4. **Begins processing** - Frames flow through the pipeline

```python
# From api/api.py - handle_start_stream()
protocol = TrickleProtocol(
    params.subscribe_url,   # Input video stream
    params.publish_url,     # Output video stream
    params.control_url,     # Parameter updates (optional)
    params.events_url,      # Monitoring events (optional)
    input_width, input_height,
    output_width, output_height,
)
streamer = PipelineStreamer(protocol, process, ...)
await streamer.start(params.params)
```

Each stream creates its own Trickle connections that are torn down when the stream ends.

---

## 🔧 Known Issues & Workarounds

### ComfyUI Shutdown Issues

**Problem**: ComfyUI pipeline has trouble shutting down cleanly, causing restarts not to recover.

**Workaround**: Skip process restart for ComfyUI, move directly to ERROR state so the worker restarts the container.

```python
if self.pipeline == "comfyui":
    raise Exception("Skipping process restart due to pipeline shutdown issues")
```

### GPU Memory in Subprocess

**Problem**: CUDA environment not inherited by spawned subprocess.

**Workaround**: Explicitly set `CUDA_VISIBLE_DEVICES` in child process:

```python
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())
```

### Queue Deadlocks

**Problem**: Closing stdout while process is writing can hang.

**Workaround**: Close in daemon thread with timeout:

```python
stdout = self.process.stdout
threading.Thread(target=lambda: stdout.close(), daemon=True).start()
```

### Tensor Format Mismatches

**Problem**: Different pipelines expect different tensor formats.

**Pattern**: Always normalize at pipeline boundaries:
- Input: `(B, H, W, C)` in range `[-1, 1]`
- Convert inside pipeline as needed
- Output: `(B, H, W, C)` for encoder

---

## 📋 Acceptance Criteria for Changes

### Adding a New Pipeline

1. Create new directory under `runner/src/runner/live/pipelines/`
2. Implement `Pipeline` ABC (initialize, put_video_frame, get_processed_video_frame, update_params, stop, prepare_models)
3. Create `*Params` class extending `BaseParams`
4. Add to `loader.py` (load_pipeline and parse_pipeline_params)
5. Add Docker configuration if needed (`docker/Dockerfile.live-app-*`)

### Modifying Process Architecture

1. Ensure graceful shutdown is preserved
2. Test restart behavior (up to 3 restarts)
3. Verify health state transitions
4. Check parent death signals work on Linux

### Modifying Streaming

1. Test stream start via API (`POST /api/live-video-to-video`)
2. Verify control messages are processed via control Trickle channel
3. Check monitoring events are emitted via events Trickle channel
4. Test stream restart (previous stream stopped before new one starts)

---

## 📚 External Resources

- [Livepeer AI Subnet Documentation](https://docs.livepeer.org)
- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Trickle Protocol](https://github.com/livepeer/go-livepeer/tree/master/trickle)

---

## 📋 Change Log

**Last Updated**: 2025-11-25

**Recent Changes**:
- Deprecated ZeroMQ protocol references (was only used for local development)
- Clarified that Trickle is the only streaming protocol
- Added "Stream Start Flow" section explaining how streams are initiated via API
- Initial AGENTS.md created for agent-assisted development
- Documented two-level process architecture
- Mapped all key files and their responsibilities

---

**Project Status**: Active development

**Maintainer Guidelines**:
- **AGENTS.md** → High-level architecture, navigation, current implementation state
- **`docs/` folder** → Detailed implementation guides
- Update this file when making architectural changes
- Keep focus on what agents need to know for safe code modifications


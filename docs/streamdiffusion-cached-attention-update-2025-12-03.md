## 2025-12-03 StreamDiffusion Cached Attention Update (PR [#860](https://github.com/livepeer/ai-runner/pull/860))

This note documents cached attention support in the StreamDiffusion pipeline.

### Schema
- `cached_attention` is a structured params block with the following fields:
  - `enabled`: boolean to enable/disable cached attention (requires reload). When enabled, reuses key/value tensors across frames to improve performance.
  - `max_frames`: number of historical frames retained in the attention cache (dynamic, clamped to [1, 4]). Limited by TensorRT engine exports.
  - `interval`: cadence in frames for refreshing the cache (dynamic, range 1–1440). Controls how often cached key/value tensors are updated.

```json
{
  "cached_attention": {
    "enabled": true,
    "max_frames": 2,
    "interval": 12
  }
}
```

### Runtime behavior
- Cached attention requires TensorRT acceleration and a 512×512 base resolution. The validator enforces the constraint before a pipeline spins up.
- Changing `enabled` triggers a full pipeline reload (and engine rebuild during `prepare_models`), but `max_frames` / `interval` are dynamic.
- `interval` is **frame-based**. It accepts integers `1–1440`, representing how many frames elapse between cache refreshes. Example: `interval=12` ≈ 0.5s at 24 FPS.
- `max_frames` and `interval` can be updated dynamically when cached attention is already enabled; toggling `enabled` requires a reload.

### Example payload
```json
{
  "model_id": "stabilityai/sd-turbo",
  "cached_attention": {
    "enabled": true,
    "max_frames": 2,
    "interval": 12
  }
}
```

## 2025-12-03 StreamDiffusion Cached Attention Update

This note captures the schema and runtime changes that ship cached attention (formerly streamv2v) support in the StreamDiffusion pipeline.

### Schema additions
- `cached_attention` is a structured params block that replaces the loose `use_cached_attn`, `cache_maxframes`, and `cache_interval` keys.
- Legacy payloads that either send the old top-level fields or the previous `stream_v2v` object are still accepted; the runner transparently folds them into the new structure for backwards compatibility.
- `min_max_frames` / `max_max_frames` describe the TensorRT profile bounds that should be compiled when preparing engines. They cannot be tuned at runtime.

```yaml
cached_attention:
  enabled: true
  max_frames: 2           # runtime-adjustable, clamped to [min_max_frames, max_max_frames]
  interval_sec: 1.0       # runtime-adjustable cadence for refreshing the cache
  min_max_frames: 1       # TensorRT profile lower bound (compile-time)
  max_max_frames: 4       # TensorRT profile upper bound (compile-time)
```

### Runtime behavior
- Cached attention requires TensorRT acceleration and a 512×512 base resolution. The validator enforces the constraint before a pipeline spins up.
- Changing `enabled`, `min_max_frames`, or `max_max_frames` triggers a full pipeline reload (and engine rebuild during `prepare_models`).
- `max_frames` and `interval_sec` can be updated dynamically; the pipeline converts `interval_sec` values into the tick-based format expected by the wrapper and clamps them to safe defaults.
- `prepare_streamdiffusion_models()` now builds engine pairs (cached attention on/off) for every model + IPAdapter combination so the worker has pre-built artifacts ready for either configuration.

### Example payload
```json
{
  "model_id": "stabilityai/sd-turbo",
  "cached_attention": {
    "enabled": true,
    "max_frames": 2,
    "interval_sec": 0.75,
    "min_max_frames": 1,
    "max_max_frames": 4
  }
}
```

## 2025-12-03 StreamDiffusion Cached Attention Update

This note captures the schema and runtime changes that ship cached attention (formerly streamv2v) support in the StreamDiffusion pipeline.

### Schema additions
- `cached_attention` is a structured params block that replaces the loose `use_cached_attn`, `cache_maxframes`, and `cache_interval` keys.
- Legacy payloads that either send the old top-level fields or the previous `stream_v2v` object are still accepted; the runner transparently folds them into the new structure for backwards compatibility.
- The max-frame bounds are fixed internally (`1–4`) and enforced by the schema; users only set the desired `max_frames` within that window.

```yaml
cached_attention:
  enabled: true
  max_frames: 2           # runtime-adjustable, clamped to [1, 4]
  interval_sec: 1.0       # runtime-adjustable cadence for refreshing the cache
```

### Runtime behavior
- Cached attention requires TensorRT acceleration and a 512×512 base resolution. The validator enforces the constraint before a pipeline spins up.
- Changing `enabled` still triggers a full pipeline reload (and engine rebuild during `prepare_models`), but `max_frames` / `interval_sec` are dynamic.
- `max_frames` and `interval_sec` can be updated dynamically; the pipeline converts `interval_sec` values into the tick-based format expected by the wrapper and clamps them to safe defaults.
- `prepare_streamdiffusion_models()` now builds engine pairs (cached attention on/off) for every model + IPAdapter combination so the worker has pre-built artifacts ready for either configuration.

### Example payload
```json
{
  "model_id": "stabilityai/sd-turbo",
  "cached_attention": {
    "enabled": true,
    "max_frames": 2,
    "interval_sec": 0.75
  }
}
```

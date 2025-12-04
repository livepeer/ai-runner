## 2025-12-03 StreamDiffusion Cached Attention Update (PR [#860](https://github.com/livepeer/ai-runner/pull/860))

This note captures the schema and runtime changes that ship cached attention (formerly streamv2v) support in the StreamDiffusion pipeline.

### Schema additions
- `cached_attention` is a structured params block that replaces the loose `use_cached_attn`, `cache_maxframes`, and `cache_interval` keys.
- Legacy payloads that either send the old top-level fields or the previous `stream_v2v` object are still accepted; the runner transparently folds them into the new structure for backwards compatibility.
- The max-frame bounds are fixed internally (`1–4`) and enforced by the schema; users only set the desired `max_frames` within that window.

```yaml
cached_attention:
  enabled: true
  max_frames: 2           # runtime-adjustable, clamped to [1, 4]
  interval: 12            # runtime-adjustable cadence in FRAMES (1–1440)
```

### Runtime behavior
- Cached attention requires TensorRT acceleration and a 512×512 base resolution. The validator enforces the constraint before a pipeline spins up.
- Changing `enabled` still triggers a full pipeline reload (and engine rebuild during `prepare_models`), but `max_frames` / `interval` are dynamic.
- `interval` is now **frame-based** (not seconds). It accepts integers `1–1440`, representing how many frames elapse between cache refreshes. Example: `interval=12` ≈ 0.5s at 24 FPS.
- `max_frames` and `interval` can be updated dynamically when cached attention is already enabled; toggling `enabled` still requires a reload.
- `prepare_streamdiffusion_models()` builds engine pairs (cached attention on/off) for every model + IPAdapter combination so the worker has pre-built artifacts ready for either configuration.

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

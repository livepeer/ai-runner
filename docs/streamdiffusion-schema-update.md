## StreamDiffusion Schema Update (PR 808)

Latest StreamDiffusion changes broaden what the Gateway can ask the runner to do. This note gives the frontend team a working mental model for the updated JSON schema and highlights the new features that landed with PR 808.

### At a Glance
- `controlnets[]` accepts TemporalNet v2 models and now exposes `conditioning_channels` for flow-based guidance.
- Four new processing blocks (`image_preprocessing`, `image_postprocessing`, `latent_preprocessing`, `latent_postprocessing`) share a single schema so the UI can build editors generically.
- New processors ship with ready-to-use defaults: TemporalNet (optical-flow ControlNet), Latent Feedback (latent-domain smoothing) and the RealESRGAN TensorRT upscaler.

---

## Updated Parameter Schema

The shape of `StreamDiffusionParams` lives in `runner/app/live/pipelines/streamdiffusion_params.py`.

```394:405:runner/app/live/pipelines/streamdiffusion_params.py
    image_preprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    image_postprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    latent_preprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None
    latent_postprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None
```

Each block is optional and uses the same structure:

```105:118:runner/app/live/pipelines/streamdiffusion_params.py
class ProcessingConfig(BaseModel, Generic[ProcessorTypeT]):
    enabled: bool = True
    processors: List[SingleProcessorConfig[ProcessorTypeT]] = []
```

```84:104:runner/app/live/pipelines/streamdiffusion_params.py
class SingleProcessorConfig(BaseModel, Generic[ProcessorTypeT]):
    type: ProcessorTypeT
    enabled: bool = True
    params: ProcessorParams = {}
```

Frontend payload expectations:

```jsonc
"image_postprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "realesrgan_trt",
      "enabled": true,
      "params": {} // optional fields are processor-specific
    }
  ]
}
```

ControlNet entries gained two TemporalNet-specific tweaks:

- `model_id`: includes the TemporalNet v2 SKUs for SD 1.5, 2.1 and SDXL.
- `conditioning_channels`: optional override (TemporalNet defaults to 6 channels).
- `preprocessor`: new values `temporal_net_tensorrt` and `feedback`.

```228:238:runner/app/live/pipelines/streamdiffusion_params.py
ControlNetConfig(
    model_id="daydreamlive/TemporalNet2-stable-diffusion-2-1",
    conditioning_scale=0.0,
    preprocessor="temporal_net_tensorrt",
    preprocessor_params={"flow_strength": 0.4},
)
```

---

## Feature Guides

### TemporalNet (optical flow ControlNet)
- **What it does:** Supplies per-frame optical-flow hints so the diffusion model aligns motion across frames, cutting flicker while keeping structure from the source stream.
- **How to enable:** Add a ControlNet entry with a TemporalNet v2 `model_id` and `preprocessor` set to `temporal_net_tensorrt`. Leave `conditioning_channels` unset to pick the default of 6.
- **Key knobs:**
  - `conditioning_scale`: start around `0.25`–`0.4`. 0 disables it, >0.6 can over-constrain motion.
  - `preprocessor_params.flow_strength`: 0.4 is a balanced default. Lower for subtle smoothing, raise (max 1.0) when the subject moves fast.
- **Runtime requirements:** TensorRT RAFT engine at `./engines/temporal_net/raft_small_min_384x384_max_1024x1024.engine` is bundled in the streamdiffusion image; nothing extra needed client-side.

Example payload fragment:

```json
"controlnets": [
  {
    "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
    "conditioning_scale": 0.35,
    "preprocessor": "temporal_net_tensorrt",
    "preprocessor_params": {
      "flow_strength": 0.4
    },
    "enabled": true
  }
]
```

### Latent Feedback
- **What it does:** Re-injects a lightly weighted version of the previous latent into the next step. This keeps color and coarse structure consistent without washing out new detail.
- **Where it runs:** Set under `latent_preprocessing` (before denoising) or `latent_postprocessing` (after denoising). Only processor name currently exposed is `latent_feedback`.
- **Usage tips:**
  - Leave `params` empty to use library defaults.
  - Enable in `latent_preprocessing` for soft persistence, or in `latent_postprocessing` when you want the final latent to blend back into the feedback loop.

Example:

```json
"latent_preprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "latent_feedback",
      "enabled": true,
      "params": {}
    }
  ]
}
```

### RealESRGAN TensorRT Upscaler
- **What it does:** Runs a 2× super-resolution pass on the generated frame using RealESRGAN optimized with TensorRT.
- **Where it runs:** Include in `image_postprocessing`.
- **Parameters:**
  - `scale_factor`: optional override. Defaults to `2.0`. `StreamDiffusionParams.get_output_resolution()` uses this to report final stream size.
- **Runtime expectations:** Requires RealESRGAN engines copied under `./models` inside the container. The stock StreamDiffusion Docker image already binds `/models/StreamDiffusion--engines/cwd_models` there.

Example:

```json
"image_postprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "realesrgan_trt",
      "enabled": true,
      "params": {
        "scale_factor": 2.0
      }
    }
  ]
}
```

---

## Processor Reference by Stage

| Stage field | Typical processors | What they are good for | Notes |
| --- | --- | --- | --- |
| `image_preprocessing` | `blur`, `canny`, `depth`, `depth_tensorrt`, `hed`, `lineart`, `mediapipe_pose`, `mediapipe_segmentation`, `openpose`, `pose_tensorrt`, `soft_edge`, `temporal_net_tensorrt` | Build conditioning inputs before diffusion. Use `pose_tensorrt`/`openpose` for pose control, `depth_tensorrt` for geometry, `canny`/`soft_edge` for edges. `temporal_net_tensorrt` only makes sense when paired with TemporalNet ControlNet. | Runs before ControlNet sampling. Avoid `realesrgan_trt` or `upscale` here. |
| `controlnets[].preprocessor` | same list as `image_preprocessing` plus `feedback`, `external`, `passthrough` | Per-ControlNet preprocessors. `feedback` feeds the previous output frame back into tile ControlNet to preserve detail. `external` lets backend provide precomputed maps. | Selecting here overrides any global `image_preprocessing`. |
| `image_postprocessing` | `realesrgan_trt`, `upscale`, `sharpen`, `blur` | Polish the generated frame. `realesrgan_trt` is the high-quality upscaler; `upscale` is a lightweight resize; `sharpen` adds local contrast. | Post processors run after the VAE decode and can change resolution. |
| `latent_preprocessing` / `latent_postprocessing` | `latent_feedback` | Temporal smoothing directly in latent space. | At present only `latent_feedback` is implemented. Enable either pre- or post- depending on where you want the blend to happen. |

When exposing the UI, surface all processor types but consider grouping them by suitability (e.g., hide `feedback` outside ControlNets, flag `realesrgan_trt` as “post-processing only”).

---

## Putting It Together

```json
{
  "model_id": "stabilityai/sd-turbo",
  "params": {
    "prompt": "cyberpunk neon city",
    "controlnets": [
      {
        "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
        "conditioning_scale": 0.3,
        "preprocessor": "temporal_net_tensorrt",
        "preprocessor_params": { "flow_strength": 0.4 },
        "enabled": true
      }
    ],
    "latent_preprocessing": {
      "enabled": true,
      "processors": [
        { "type": "latent_feedback", "enabled": true, "params": {} }
      ]
    },
    "image_postprocessing": {
      "enabled": true,
      "processors": [
        { "type": "realesrgan_trt", "enabled": true, "params": { "scale_factor": 2.0 } }
      ]
    }
  }
}
```

This schema works both for job creation and live updates (fields listed as dynamically updateable in `StreamDiffusionParams` do not require a pipeline reload).

---

### Quick Validation Checklist for Frontend
- [ ] For TemporalNet, ensure the selected diffusion base (`model_id`) matches the TemporalNet SKU variant (SD 1.5 vs SDXL).
- [ ] When enabling RealESRGAN, update any UI resolution badges with `get_output_resolution`.
- [ ] If multiple ControlNets target SDXL, keep the enabled count ≤3 to avoid VRAM errors.
- [ ] Send empty `params` objects instead of omitting keys when toggling processors on/off; the backend treats missing blocks as “use defaults”.


## StreamDiffusion Schema Update (PR [#808](https://github.com/livepeer/ai-runner/pull/808))

PR #808 expands what the StreamDiffusion pipeline can do without a reload. This document captures the schema additions and how to exercise the new processors and execution modes from client requests.

---

### Schema Changes at a Glance
- Four processor blocks now exist on `StreamDiffusionParams`: `image_preprocessing`, `image_postprocessing`, `latent_preprocessing`, `latent_postprocessing`. Each block shares the same shape:

  ```84:118:runner/app/live/pipelines/streamdiffusion_params.py
  class SingleProcessorConfig(BaseModel, Generic[ProcessorTypeT]):
      type: ProcessorTypeT
      enabled: bool = True
      params: ProcessorParams = {}

  class ProcessingConfig(BaseModel, Generic[ProcessorTypeT]):
      enabled: bool = True
      processors: List[SingleProcessorConfig[ProcessorTypeT]] = []
  ```

- ControlNets gained support for the TemporalNet v2 models, plus a backend-populated `conditioning_channels` field. **Clients must omit `conditioning_channels`**—the runner fills it (6 channels for TemporalNet, 3 for the rest) and ignores user-supplied values.
- `skip_diffusion` is now exposed so a request can run just the processors (e.g. stream live depth, run a pure upscale) without invoking the diffusion/denoising step.

TemporalNet requires a ControlNet that matches the chosen diffusion base. The table below lists the valid pairings:

| Diffusion `model_id`                | Model type | Matching TemporalNet ControlNet                                       |
| ----------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `stabilityai/sd-turbo`              | sd21       | `daydreamlive/TemporalNet2-stable-diffusion-2-1`                       |
| `prompthero/openjourney-v4`         | sd15       | `daydreamlive/TemporalNet2-stable-diffusion-v1-5`                      |
| `Lykon/dreamshaper-8`              | sd15       | `daydreamlive/TemporalNet2-stable-diffusion-v1-5`                      |
| `stabilityai/sdxl-turbo`            | sdxl       | `daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0`               |

---

## Feature Details

### TemporalNet ControlNet (`temporal_net_tensorrt`)
- Supplies frame-to-frame optical flow so diffusion sticks to motion from the source stream.
- Set the ControlNet’s `preprocessor` to `temporal_net_tensorrt`. Leave `conditioning_channels` absent.
- **`conditioning_scale` guidance:** values in the `0.25–0.4` band preserve structure while leaving room for creative edits. Pushing beyond ~0.6 drastically suppresses the “AI” effect—you usually end up shrinking `t_index_list` to maintain stability, often with diminishing returns.
- **`preprocessor_params.flow_strength`** follows the upstream definition: “Strength multiplier for optical flow visualization (1.0 = normal, higher = more pronounced flow)”. Lower values mute the motion field, higher values exaggerate it.

```json
"controlnets": [
  {
    "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
    "conditioning_scale": 0.35,
    "preprocessor": "temporal_net_tensorrt",
    "preprocessor_params": {
      "flow_strength": 1.0
    },
    "enabled": true
  }
]
```

> ❗️ **Do not send `conditioning_channels`.** The runner derives it automatically (6 for TemporalNet, 3 otherwise) and will ignore the client value.

### Latent Feedback Processor (`latent_feedback`)
Latent feedback blends the previous latent back into the new latent before diffusion, giving temporal consistency with minimal cost:

```
output_latent = (1 - feedback_strength) * input_latent + feedback_strength * previous_latent
```

- Available only under `latent_preprocessing`.
- `feedback_strength` controls how much of the previous frame’s latent leaks in:
  - `0.0`: pass-through (no feedback)
  - `0.5`: 50/50 blend (default)
  - `1.0`: reuse the previous latent entirely
- On the first frame (no cached latent) the preprocessor falls back to the input latent.

```json
"latent_preprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "latent_feedback",
      "enabled": true,
      "params": {
        "feedback_strength": 0.5
      }
    }
  ]
}
```

### Image Post-Processing & RealESRGAN (`realesrgan_trt`)
- Runs after the diffusion step to enhance decoded frames (default 2× super resolution).
- Provide optional `scale_factor` in `params`.
- `StreamDiffusionParams.get_output_resolution()` multiplies the base width/height by `scale_factor` for each enabled upscaler.
- ⚠️ **Cannot be toggled mid-stream.** Enabling or disabling RealESRGAN requires a full pipeline restart because resolution changes break the streaming stack. Choose the final upscale strategy before calling the pipeline.

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

### Processor Catalog

| Stage field                | Typical processors                                                                                                                                         | Notes                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `image_preprocessing`      | `blur`, `canny`, `depth`, `depth_tensorrt`, `hed`, `lineart`, `mediapipe_pose`, `mediapipe_segmentation`, `openpose`, `pose_tensorrt`, `soft_edge`, `temporal_net_tensorrt` | Produces conditioning inputs before diffusion. Only use `temporal_net_tensorrt` with TemporalNet ControlNet entries. |
| `controlnets[].preprocessor` | Same list as `image_preprocessing` plus `feedback`, `external`, `passthrough`                                                                              | Overrides per-ControlNet preprocessing. `feedback` pipes the previous frame into tile ControlNet. |
| `image_postprocessing`     | `realesrgan_trt`, `upscale`, `sharpen`, `blur`                                                                                                             | Runs on decoded images. `realesrgan_trt` and `upscale` change output resolution.                 |
| `latent_preprocessing`     | `latent_feedback`                                                                                                                                          | Only latent processor currently available.                                                       |

### `skip_diffusion`
Setting `skip_diffusion: true` skips VAE encode → diffusion → decode while still running the configured pre/post-processors. Example use cases:
- stream the output of a preprocessor (e.g. live depth maps or pose skeletons),
- run post-processors like RealESRGAN on externally provided frames,
- warm a pipeline without paying the diffusion cost.

```json
{
  "model_id": "stabilityai/sd-turbo",
  "skip_diffusion": true,
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      { "type": "realesrgan_trt", "enabled": true, "params": { "scale_factor": 2.0 } }
    ]
  }
}
```

Any ControlNets or diffusion-only parameters are ignored when `skip_diffusion` is enabled.

---

## Request Examples

### SD 2.1 (Turbo) with TemporalNet, latent feedback, RealESRGAN

```json
{
  "model_id": "stabilityai/sd-turbo",
  "prompt": "cyberpunk neon city",
  "negative_prompt": "blurry, low quality",
  "skip_diffusion": false,
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.32,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": { "flow_strength": 1.0 },
      "enabled": true
    }
  ],
  "latent_preprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "latent_feedback",
        "enabled": true,
        "params": { "feedback_strength": 0.4 }
      }
    ]
  },
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "realesrgan_trt",
        "enabled": true,
        "params": { "scale_factor": 2.0 }
      }
    ]
  }
}
```

### SDXL Turbo with TemporalNet only

```json
{
  "model_id": "stabilityai/sdxl-turbo",
  "prompt": "studio portrait in neon lighting",
  "skip_diffusion": false,
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0",
      "conditioning_scale": 0.28,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": { "flow_strength": 0.8 },
      "enabled": true
    }
  ]
}
```

> ℹ️ SDXL requests can enable at most three ControlNets at once (hardware limit enforced by the runner).

### SD 1.5 (DreamShaper) with latent-only workflow (`skip_diffusion`)

```json
{
  "model_id": "Lykon/dreamshaper-8",
  "skip_diffusion": true,
  "latent_preprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "latent_feedback",
        "enabled": true,
        "params": { "feedback_strength": 0.5 }
      }
    ]
  }
}
```

---

Most fields shown above can be updated during a stream; notable exceptions:
- RealESRGAN (`realesrgan_trt`) requires a restart to change because it affects resolution.
- `skip_diffusion` is evaluated at pipeline creation time; switching modes mid-stream triggers a reload.

Refer back to `StreamDiffusionParams` for the authoritative list of runtime-updateable attributes.

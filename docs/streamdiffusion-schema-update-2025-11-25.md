## 2025-11-25 StreamDiffusion Schema Update (PR [#808](https://github.com/livepeer/ai-runner/pull/808))

PR #808 expands what the StreamDiffusion pipeline can do. This document captures the schema additions and how to exercise the new processors and execution modes from the params schema.

---

### Schema Changes at a Glance
- Four new "processor" blocks now exist on the pipeline params: `image_preprocessing`, `image_postprocessing`, `latent_preprocessing`, `latent_postprocessing`. Each block shares the same `ProcessingConfig` shape:

  ```84:118:runner/app/live/pipelines/streamdiffusion/params.py
  class SingleProcessorConfig(BaseModel, Generic[ProcessorTypeT]):
      type: ProcessorTypeT
      enabled: bool = True
      params: ProcessorParams = {}

  class ProcessingConfig(BaseModel, Generic[ProcessorTypeT]):
      enabled: bool = True
      processors: List[SingleProcessorConfig[ProcessorTypeT]] = []
  ```

- ControlNets gained support for the TemporalNet v2 models, plus a backend-populated `conditioning_channels` field. **Clients should omit `conditioning_channels`**—they are well-defined per model and the runner fills it automatically (6 channels for TemporalNet, 3 for the rest).
- `skip_diffusion` is now exposed so a request can run just the processors (e.g. depth-only stream, run a pure upscale) without invoking the diffusion/denoising step.

---

## Feature Details

### TemporalNet ControlNet (`temporal_net_tensorrt`)
- Supplies frame-to-frame optical flow so diffusion sticks to motion from the source stream.
- Set the ControlNet’s `preprocessor` to `temporal_net_tensorrt`. Leave `conditioning_channels` absent.
- **`conditioning_scale` guidance:** values in the `0.25–0.4` band preserve structure while leaving room for creative edits. Pushing beyond ~0.6 drastically suppresses the “AI effect"—you usually end up having to shrink `t_index_list` to get the trippiness back, often with diminishing returns. It can go higher than 1 for even stronger conditioning.
- **`preprocessor_params.flow_strength`**: Strength multiplier for optical flow visualization (1.0 = normal, higher = more pronounced flow). Lower values mute the motion field, higher values exaggerate it.

e.g.:
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

> ❗️ **Do not send `conditioning_channels`.** The runner derives it automatically (6 for TemporalNet, 3 otherwise).

The TemporalNet controlnet has different models, one for each U-net model type. The table below lists the valid pairings:

| Diffusion `model_id`                | Model type | Matching TemporalNet ControlNet                                       |
| ----------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `stabilityai/sd-turbo`              | sd21       | `daydreamlive/TemporalNet2-stable-diffusion-2-1`                       |
| `prompthero/openjourney-v4`         | sd15       | `daydreamlive/TemporalNet2-stable-diffusion-v1-5`                      |
| `Lykon/dreamshaper-8`              | sd15       | `daydreamlive/TemporalNet2-stable-diffusion-v1-5`                      |
| `stabilityai/sdxl-turbo`            | sdxl       | `daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0`               |


### Latent Feedback Processor (`latent_feedback`)
Latent feedback blends the previous latent back into the new latent before diffusion, giving temporal consistency with minimal cost:

```
output_latent = (1 - feedback_strength) * input_latent + feedback_strength * previous_latent
```

- Available under `latent_preprocessing` (before denoising) or `latent_postprocessing` (after denoising).
  - Enable in `latent_preprocessing` to mix the previous input frame with the current one and preserve motion (similar to temporalnet optical flow), or in `latent_postprocessing` to blend the final output latent with the previous output, improving consistency.
- `feedback_strength` controls how much of the previous frame’s latent leaks in:
  - `0.0`: pass-through (no feedback)
  - `0.15`: Default value, provides some nice feedback
  - `0.4`: Maximum reasonable value that doesn't overshoot into garbage output
  - `1.0`: reuse the previous latent entirely
- On the first frame (no cached latent) the preprocessor falls back to the input latent.

e.g.:
```json
"latent_preprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "latent_feedback",
      "enabled": true,
      "params": {
        "feedback_strength": 0.15
      }
    }
  ]
}
```
(notice that all `enabled` fields are optional and can be ommited)

### Image Post-Processing & RealESRGAN (`realesrgan_trt`)
- Runs after the diffusion step to enhance decoded frames for 2× super resolution.
- ⚠️ **Cannot be toggled mid-stream.** Enabling or disabling RealESRGAN requires a full pipeline restart because resolution changes break the streaming stack. When changing upscaler processors, must completely restart the stream.

e.g.:
```json
"image_postprocessing": {
  "enabled": true,
  "processors": [
    {
      "type": "realesrgan_trt",
      "enabled": true,
    }
  ]
}
```
(notice that all `enabled` fields are optional and can be ommited)

### Processor Catalog

Additionally from the above, there are still other processors that can still be used. For detailed documentation check the corresponding classes and metadata in the [StreamDiffusion `processors` folder](https://github.com/livepeer/StreamDiffusion/tree/main/src/streamdiffusion/preprocessing/processors).

These processors have specific domains, and can only be used on either `image_` or `latent_` processors configs. The controlnet preprocessors are the same as the `image_` processors. Any preprocessor can also be used as a postprocessor (and the other way around), but there are typical places where they are usually used, mapped below.

| Stage field                | Typical processors                                                                                                                                         | Notes                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `image_preprocessing`      | `blur`, `canny`, `depth`, `depth_tensorrt`, `hed`, `lineart`, `mediapipe_pose`, `mediapipe_segmentation`, `openpose`, `pose_tensorrt`, `soft_edge`, `temporal_net_tensorrt` | Produces conditioning inputs before diffusion. Only use `temporal_net_tensorrt` with TemporalNet ControlNet entries. |
| `controlnets[].preprocessor` | Same list as `image_preprocessing` plus `feedback`, `external`, `passthrough`                                                                              | Overrides per-ControlNet preprocessing. `feedback` pipes the previous frame into tile ControlNet. |
| `image_postprocessing`     | `realesrgan_trt`, `upscale`, `sharpen`, `blur`                                                                                                             | Runs on decoded images. `realesrgan_trt` and `upscale` change output resolution.                 |
| `latent_preprocessing`     | `latent_feedback`                                                                                                                                          | Only latent processor currently available.                                                       |

### `skip_diffusion`
Setting `skip_diffusion: true` skips VAE encode → diffusion → decode while still running the configured pre/post-processors. Example use cases:
- stream the output of a preprocessor (e.g. live depth maps or pose skeletons),
- run post-processors like RealESRGAN upscaler on externally provided frames,
- warm a pipeline without paying the diffusion cost.

e.g.:
```json
{
  "skip_diffusion": true,
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      { "type": "realesrgan_trt", "enabled": true }
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
        "params": { "feedback_strength": 0.2 }
      }
    ]
  },
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "realesrgan_trt",
        "enabled": true
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
- `skip_diffusion` is evaluated at pipeline creation time; switching modes mid-stream triggers a pipeline reload.

Refer back to [`StreamDiffusionParams`](https://github.com/livepeer/ai-runner/blob/main/runner/app/live/pipelines/streamdiffusion/params.py) for the authoritative list of runtime-updateable attributes.

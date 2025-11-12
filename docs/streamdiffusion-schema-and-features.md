# StreamDiffusion Schema Changes and New Features

This document describes the schema changes and new features introduced in PR #808 for the StreamDiffusion pipeline, specifically for frontend developers integrating with the Livepeer AI Runner API.

## Table of Contents

- [Overview](#overview)
- [Schema Changes](#schema-changes)
  - [Processors Architecture](#processors-architecture)
  - [New Fields](#new-fields)
- [New Features](#new-features)
  - [TemporalNet](#temporalnet)
  - [Latent Feedback](#latent-feedback)
  - [RealESRGAN Upscaler](#realesrgan-upscaler)
- [Processors Guide](#processors-guide)
  - [Image Preprocessing Processors](#image-preprocessing-processors)
  - [Image Postprocessing Processors](#image-postprocessing-processors)
  - [Latent Preprocessing Processors](#latent-preprocessing-processors)
  - [Latent Postprocessing Processors](#latent-postprocessing-processors)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)

## Overview

The StreamDiffusion pipeline now supports a flexible processor system that allows you to apply image and latent transformations at different stages of the pipeline. Additionally, three major features have been added:

1. **TemporalNet**: A ControlNet model for temporal consistency in video generation
2. **Latent Feedback**: A processor that feeds previous frame information back into the pipeline
3. **RealESRGAN Upscaler**: A high-quality image upscaling processor

## Schema Changes

### Processors Architecture

The pipeline now uses a generic processor system with four distinct processing stages:

1. **Image Preprocessing** (`image_preprocessing`): Applied to input frames before they enter the diffusion pipeline
2. **Image Postprocessing** (`image_postprocessing`): Applied to output frames after diffusion
3. **Latent Preprocessing** (`latent_preprocessing`): Applied to latent representations before diffusion steps
4. **Latent Postprocessing** (`latent_postprocessing`): Applied to latent representations after diffusion steps

Each processing stage follows this structure:

```typescript
{
  enabled: boolean,           // Whether this processing stage is active
  processors: [               // List of processors to apply in order
    {
      type: string,           // Processor type (see below)
      enabled: boolean,       // Whether this specific processor is active
      params: {               // Processor-specific parameters
        // ... varies by processor type
      }
    }
  ]
}
```

### New Fields

The following fields have been added to `StreamDiffusionParams`:

| Field | Type | Description |
|-------|------|-------------|
| `image_preprocessing` | `ProcessingConfig<ImageProcessorName>` \| `null` | Image processors applied before diffusion |
| `image_postprocessing` | `ProcessingConfig<ImageProcessorName>` \| `null` | Image processors applied after diffusion |
| `latent_preprocessing` | `ProcessingConfig<LatentProcessorsName>` \| `null` | Latent processors applied before diffusion steps |
| `latent_postprocessing` | `ProcessingConfig<LatentProcessorsName>` \| `null` | Latent processors applied after diffusion steps |

Additionally, `ControlNetConfig` has been updated:

| Field | Type | Description |
|-------|------|-------------|
| `conditioning_channels` | `int` \| `null` | Number of channels in the ControlNet conditioning input. Defaults to 6 for TemporalNets, 3 for others |

## New Features

### TemporalNet

TemporalNet is a ControlNet model designed to improve temporal consistency in video generation by using optical flow information from previous frames.

#### Supported Models

- **SD21 (SD-Turbo)**: `daydreamlive/TemporalNet2-stable-diffusion-2-1`
- **SD15**: `daydreamlive/TemporalNet2-stable-diffusion-v1-5`
- **SDXL**: `daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0`

#### Configuration

TemporalNet is configured as a ControlNet with the `temporal_net_tensorrt` preprocessor:

```json
{
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.4,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": {
        "flow_strength": 0.4
      },
      "enabled": true,
      "control_guidance_start": 0.0,
      "control_guidance_end": 1.0
    }
  ]
}
```

#### Parameters

- **`flow_strength`** (float, default: 0.4): Controls the strength of optical flow guidance. Higher values increase temporal consistency but may reduce creative variation.
- **`conditioning_scale`** (float): Standard ControlNet conditioning scale. Typical range: 0.0-1.0. Start with 0.4-0.6 for balanced results.

#### Usage Tips

- TemporalNet works best when `conditioning_scale` is set between 0.3-0.6
- Lower `flow_strength` values (0.2-0.4) provide smoother transitions with less rigidity
- Higher `flow_strength` values (0.5-0.8) provide stronger temporal consistency but may reduce motion
- TemporalNet requires TensorRT acceleration and the RAFT engine to be compiled

### Latent Feedback

Latent feedback is a latent processor that feeds information from previous frames back into the diffusion process, helping maintain consistency across frames.

#### Configuration

Latent feedback is configured in the `latent_preprocessing` or `latent_postprocessing` stage:

```json
{
  "latent_preprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "latent_feedback",
        "enabled": true,
        "params": {
          // Processor-specific parameters
        }
      }
    ]
  }
}
```

#### When to Use

- **Latent Preprocessing**: Use when you want to influence the diffusion process from the start
- **Latent Postprocessing**: Use when you want to refine the output based on previous frames

#### Usage Tips

- Latent feedback is particularly useful for maintaining style consistency across frames
- Combine with TemporalNet for maximum temporal stability
- Experiment with different feedback strengths through processor parameters

### RealESRGAN Upscaler

RealESRGAN is a high-quality image upscaling processor that can double the resolution of output frames using TensorRT acceleration.

#### Configuration

RealESRGAN is configured as an image postprocessing processor:

```json
{
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "realesrgan_trt",
        "enabled": true,
        "params": {}
      }
    ]
  }
}
```

#### Behavior

- **Fixed 2x upscaling**: RealESRGAN always upscales by a factor of 2 (e.g., 512x512 → 1024x1024)
- **Automatic resolution calculation**: The pipeline automatically calculates output resolution when upscalers are present
- **TensorRT acceleration**: Requires TensorRT engine compilation for optimal performance

#### Alternative: Generic Upscaler

For custom scale factors, use the generic `upscale` processor:

```json
{
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "upscale",
        "enabled": true,
        "params": {
          "scale_factor": 2.0  // Custom scale factor
        }
      }
    ]
  }
}
```

#### Output Resolution

When upscalers are enabled, the output resolution is automatically calculated. Use the `get_output_resolution()` method (or equivalent API endpoint) to determine the actual output dimensions.

#### Important Notes

- **Resolution changes mid-stream**: Output resolution changes (e.g., enabling/disabling upscalers) are **not allowed** during an active stream. You must start a new stream to change resolution.
- **Performance**: RealESRGAN adds processing time but provides superior quality compared to simple upscaling methods

## Processors Guide

### Processor Types Overview

The processor system is designed to be flexible and extensible. However, not all processors are useful in every stage of the pipeline. Below is a guide to which processors are most useful in each stage.

### Image Preprocessing Processors

Image preprocessing processors operate on input frames **before** they enter the diffusion pipeline. These are typically used for:

- Preparing frames for ControlNet conditioning
- Applying filters or transformations to input
- Normalizing or adjusting input images

#### Useful Processors

| Processor | Use Case | Notes |
|-----------|----------|-------|
| `passthrough` | No preprocessing needed | Default for most ControlNets |
| `canny` | Edge detection for Canny ControlNet | Use with canny ControlNet models |
| `depth_tensorrt` | Depth estimation | Use with depth ControlNet models |
| `pose_tensorrt` | Pose estimation | Use with openpose ControlNet models |
| `temporal_net_tensorrt` | Optical flow for TemporalNet | **Required** for TemporalNet ControlNet |
| `soft_edge` | Soft edge detection | Use with HED ControlNet models |
| `feedback` | Previous frame feedback | Useful for maintaining consistency |

#### Less Common in Preprocessing

- `realesrgan_trt`, `upscale`: These are typically postprocessing operations
- `blur`, `sharpen`: Usually applied as postprocessing effects

### Image Postprocessing Processors

Image postprocessing processors operate on output frames **after** diffusion. These are typically used for:

- Upscaling final output
- Applying visual effects
- Quality enhancement

#### Useful Processors

| Processor | Use Case | Notes |
|-----------|----------|-------|
| `realesrgan_trt` | High-quality 2x upscaling | **Recommended** for production upscaling |
| `upscale` | Generic upscaling | Use for custom scale factors |
| `blur` | Blur effect | Artistic effects |
| `sharpen` | Sharpening | Quality enhancement |
| `feedback` | Frame feedback | For next frame processing |

#### Less Common in Postprocessing

- `canny`, `depth_tensorrt`, `pose_tensorrt`: These are preprocessing operations for ControlNets
- `temporal_net_tensorrt`: Used in preprocessing for TemporalNet

### Latent Preprocessing Processors

Latent preprocessing processors operate on latent representations **before** diffusion steps. These are used for:

- Influencing the diffusion process from the start
- Applying transformations to latent space

#### Available Processors

| Processor | Use Case | Notes |
|-----------|----------|-------|
| `latent_feedback` | Previous frame feedback | Maintains consistency across frames |

### Latent Postprocessing Processors

Latent postprocessing processors operate on latent representations **after** diffusion steps. These are used for:

- Refining latent representations
- Applying post-diffusion transformations

#### Available Processors

| Processor | Use Case | Notes |
|-----------|----------|-------|
| `latent_feedback` | Frame refinement | Refines output based on previous frames |

### Processor Generality

While the processor system is designed to be general-purpose, in practice:

1. **Image processors** (`ImageProcessorName`) are primarily used in:
   - `image_preprocessing`: For ControlNet preprocessors and input transformations
   - `image_postprocessing`: For upscaling and visual effects

2. **Latent processors** (`LatentProcessorsName`) are used in:
   - `latent_preprocessing`: For influencing diffusion from the start
   - `latent_postprocessing`: For refining diffusion output

3. **ControlNet preprocessors** are specified in `ControlNetConfig.preprocessor`, not in the general processor lists. However, some processors (like `feedback`) can be used both as ControlNet preprocessors and as general image processors.

## Usage Examples

### Example 1: TemporalNet with RealESRGAN Upscaling

```json
{
  "model_id": "stabilityai/sd-turbo",
  "width": 512,
  "height": 512,
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.5,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": {
        "flow_strength": 0.4
      },
      "enabled": true
    }
  ],
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "realesrgan_trt",
        "enabled": true,
        "params": {}
      }
    ]
  }
}
```

**Output**: 1024x1024 frames with temporal consistency

### Example 2: Latent Feedback for Style Consistency

```json
{
  "model_id": "stabilityai/sdxl-turbo",
  "width": 512,
  "height": 512,
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
}
```

### Example 3: Multiple Postprocessing Effects

```json
{
  "image_postprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "realesrgan_trt",
        "enabled": true,
        "params": {}
      },
      {
        "type": "sharpen",
        "enabled": true,
        "params": {
          "strength": 0.5
        }
      }
    ]
  }
}
```

**Note**: Processors are applied in order, so upscaling happens first, then sharpening.

### Example 4: TemporalNet with Feedback Preprocessing

```json
{
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.4,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": {
        "flow_strength": 0.3
      },
      "enabled": true
    }
  ],
  "image_preprocessing": {
    "enabled": true,
    "processors": [
      {
        "type": "feedback",
        "enabled": true,
        "params": {}
      }
    ]
  }
}
```

## Migration Guide

### For Existing Integrations

If you're updating from a previous version:

1. **No breaking changes**: Existing API calls without processor fields will continue to work
2. **Optional fields**: All new processor fields are optional (`null` by default)
3. **ControlNet changes**: TemporalNet ControlNets are automatically available if you include them in the `controlnets` array

### Adding Processors

To add processors to an existing integration:

1. **Start with defaults**: Begin with `enabled: false` or omit the fields entirely
2. **Enable gradually**: Enable one processor at a time to understand its effects
3. **Check output resolution**: If using upscalers, ensure your decoder/display can handle the increased resolution

### Common Patterns

**Pattern 1: Upscaling Only**
```json
{
  "image_postprocessing": {
    "enabled": true,
    "processors": [{"type": "realesrgan_trt", "enabled": true, "params": {}}]
  }
}
```

**Pattern 2: Temporal Consistency**
```json
{
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.5,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": {"flow_strength": 0.4},
      "enabled": true
    }
  ]
}
```

**Pattern 3: Maximum Quality**
```json
{
  "controlnets": [
    {
      "model_id": "daydreamlive/TemporalNet2-stable-diffusion-2-1",
      "conditioning_scale": 0.5,
      "preprocessor": "temporal_net_tensorrt",
      "preprocessor_params": {"flow_strength": 0.4},
      "enabled": true
    }
  ],
  "latent_preprocessing": {
    "enabled": true,
    "processors": [{"type": "latent_feedback", "enabled": true, "params": {}}]
  },
  "image_postprocessing": {
    "enabled": true,
    "processors": [{"type": "realesrgan_trt", "enabled": true, "params": {}}]
  }
}
```

## API Considerations

### Dynamic Updates

All processor configurations (`image_preprocessing`, `image_postprocessing`, `latent_preprocessing`, `latent_postprocessing`) can be updated **dynamically** without reloading the pipeline. However:

- **Resolution changes**: Changing upscaler configuration (which changes output resolution) requires starting a new stream
- **Performance**: Some processors add latency; test performance impact before enabling in production

### Validation

The API validates:
- Processor types must match the stage (image vs latent processors)
- Enabled processors must have valid types
- Output resolution changes are blocked mid-stream

### Error Handling

Common errors:
- **Invalid processor type**: Ensure processor type matches the stage (e.g., `latent_feedback` only in latent stages)
- **Missing engine**: TemporalNet and RealESRGAN require TensorRT engines; ensure they're compiled
- **Resolution mismatch**: If upscalers are enabled, ensure your output handling supports the increased resolution

## Additional Resources

- [StreamDiffusion Parameters Reference](../runner/app/live/pipelines/streamdiffusion_params.py)
- [Default Configuration Examples](../runner/app/live/pipelines/streamdiffusion_sd15_default_params.json)
- [Development Guide](./development-guide.md)

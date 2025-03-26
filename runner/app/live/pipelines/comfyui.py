import os
import json
import torch
import asyncio
import numpy as np
from PIL import Image
from typing import Union
from pydantic import BaseModel, field_validator

from .interface import Pipeline
from comfystream.client import ComfyStreamClient
from trickle import VideoFrame, VideoOutput

import logging

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
WARMUP_RUNS = 1
DEFAULT_WORKFLOW_JSON = json.loads("""
{
  "1": {
    "inputs": {
      "image": "example.png",
      "upload": "image"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "Load Image"
    }
  },
  "2": {
    "inputs": {
      "engine": "depth_anything_vitl14-fp16.engine",
      "images": [
        "1",
        0
      ]
    },
    "class_type": "DepthAnythingTensorrt",
    "_meta": {
      "title": "Depth Anything Tensorrt"
    }
  },
  "3": {
    "inputs": {
      "unet_name": "static-dreamshaper8_SD15_$stat-b-1-h-512-w-512_00001_.engine",
      "model_type": "SD15"
    },
    "class_type": "TensorRTLoader",
    "_meta": {
      "title": "TensorRT Loader"
    }
  },
  "5": {
    "inputs": {
      "text": "the hulk",
      "clip": [
        "23",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "6": {
    "inputs": {
      "text": "",
      "clip": [
        "23",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "seed": 905056445574169,
      "steps": 1,
      "cfg": 1,
      "sampler_name": "lcm",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "3",
        0
      ],
      "positive": [
        "9",
        0
      ],
      "negative": [
        "9",
        1
      ],
      "latent_image": [
        "16",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "8": {
    "inputs": {
      "control_net_name": "control_v11f1p_sd15_depth_fp16.safetensors"
    },
    "class_type": "ControlNetLoader",
    "_meta": {
      "title": "Load ControlNet Model"
    }
  },
  "9": {
    "inputs": {
      "strength": 1,
      "start_percent": 0,
      "end_percent": 1,
      "positive": [
        "5",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "control_net": [
        "10",
        0
      ],
      "image": [
        "2",
        0
      ]
    },
    "class_type": "ControlNetApplyAdvanced",
    "_meta": {
      "title": "Apply ControlNet"
    }
  },
  "10": {
    "inputs": {
      "backend": "inductor",
      "fullgraph": false,
      "mode": "reduce-overhead",
      "controlnet": [
        "8",
        0
      ]
    },
    "class_type": "TorchCompileLoadControlNet",
    "_meta": {
      "title": "TorchCompileLoadControlNet"
    }
  },
  "11": {
    "inputs": {
      "vae_name": "taesd"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "13": {
    "inputs": {
      "backend": "inductor",
      "fullgraph": true,
      "mode": "reduce-overhead",
      "compile_encoder": true,
      "compile_decoder": true,
      "vae": [
        "11",
        0
      ]
    },
    "class_type": "TorchCompileLoadVAE",
    "_meta": {
      "title": "TorchCompileLoadVAE"
    }
  },
  "14": {
    "inputs": {
      "samples": [
        "7",
        0
      ],
      "vae": [
        "13",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "15": {
    "inputs": {
      "images": [
        "14",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "16": {
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "23": {
    "inputs": {
      "clip_name": "CLIPText/model.fp16.safetensors",
      "type": "stable_diffusion",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  }
}
""")


class ComfyUIParams(BaseModel):
    class Config:
        extra = "forbid"

    prompt: Union[str, dict] = DEFAULT_WORKFLOW_JSON

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v) -> dict:
        if v == "":
            return DEFAULT_WORKFLOW_JSON

        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed prompt JSON must be a dictionary/object")
                return parsed
            except json.JSONDecodeError:
                raise ValueError("Provided prompt string must be valid JSON")

        raise ValueError("Prompt must be either a JSON object or such JSON object serialized as a string")


class ComfyUI(Pipeline):
    def __init__(self):
        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams
        self.video_incoming_frames = asyncio.Queue()

    async def warm_video(self):
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.processed_input = torch.randn(1, 512, 512, 3)

        for _ in range(WARMUP_RUNS):
          self.client.put_video_input(dummy_frame)
          _ = await self.client.get_video_output()
        logging.info("Video frame warmup done")

    async def put_video_frame(self, frame: VideoFrame):
        image_np = np.array(frame.image.convert("RGB")).astype(np.float32) / 255.0
        frame.side_data.processed_input = torch.tensor(image_np).unsqueeze(0)
        frame.side_data.skipped = True
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(frame)

    async def get_processed_video_frame(self):
        result_tensor = await self.client.get_video_output()
        frame = await self.video_incoming_frames.get()
        while frame.side_data.skipped:
            frame = await self.video_incoming_frames.get()

        result_tensor = result_tensor.squeeze(0)
        result_image_np = (result_tensor * 255).byte()
        result_image = Image.fromarray(result_image_np.cpu().numpy())
        return VideoOutput(frame.replace_image(result_image))
    
    async def set_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Setting ComfyUI Pipeline Prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.update_prompts([new_params.prompt])
        self.params = new_params

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.client.stop()
        logging.info("ComfyUI pipeline stopped")

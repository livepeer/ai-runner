import os
import json
import torch
from PIL import Image
import asyncio
import numpy as np
from typing import Union
from pydantic import BaseModel, field_validator

from .interface import Pipeline
from comfystream.client import ComfyStreamClient

import logging

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
DEFAULT_WORKFLOW_JSON = json.loads("""
{
  "1": {
    "_meta": {
      "title": "Load Image"
    },
    "inputs": {
      "image": "example.png",
      "upload": "image"
    },
    "class_type": "LoadImage"
  },
  "2": {
    "_meta": {
      "title": "Depth Anything Tensorrt"
    },
    "inputs": {
      "engine": "depth_anything_vitl14-fp16.engine",
      "images": [
        "1",
        0
      ]
    },
    "class_type": "DepthAnythingTensorrt"
  },
  "3": {
    "_meta": {
      "title": "TensorRT Loader"
    },
    "inputs": {
      "unet_name": "static-dreamshaper8_SD15_$stat-b-1-h-512-w-512_00001_.engine",
      "model_type": "SD15"
    },
    "class_type": "TensorRTLoader"
  },
  "5": {
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    },
    "inputs": {
      "clip": [
        "23",
        0
      ],
      "text": "the hulk"
    },
    "class_type": "CLIPTextEncode"
  },
  "6": {
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    },
    "inputs": {
      "clip": [
        "23",
        0
      ],
      "text": ""
    },
    "class_type": "CLIPTextEncode"
  },
  "7": {
    "_meta": {
      "title": "KSampler"
    },
    "inputs": {
      "cfg": 1,
      "seed": 905056445574169,
      "model": [
        "3",
        0
      ],
      "steps": 1,
      "denoise": 1,
      "negative": [
        "9",
        1
      ],
      "positive": [
        "9",
        0
      ],
      "scheduler": "normal",
      "latent_image": [
        "16",
        0
      ],
      "sampler_name": "lcm"
    },
    "class_type": "KSampler"
  },
  "8": {
    "_meta": {
      "title": "Load ControlNet Model"
    },
    "inputs": {
      "control_net_name": "control_v11f1p_sd15_depth_fp16.safetensors"
    },
    "class_type": "ControlNetLoader"
  },
  "9": {
    "_meta": {
      "title": "Apply ControlNet"
    },
    "inputs": {
      "image": [
        "2",
        0
      ],
      "negative": [
        "6",
        0
      ],
      "positive": [
        "5",
        0
      ],
      "strength": 1,
      "control_net": [
        "10",
        0
      ],
      "end_percent": 1,
      "start_percent": 0
    },
    "class_type": "ControlNetApplyAdvanced"
  },
  "10": {
    "_meta": {
      "title": "TorchCompileLoadControlNet"
    },
    "inputs": {
      "mode": "reduce-overhead",
      "backend": "inductor",
      "fullgraph": false,
      "controlnet": [
        "8",
        0
      ]
    },
    "class_type": "TorchCompileLoadControlNet"
  },
  "11": {
    "_meta": {
      "title": "Load VAE"
    },
    "inputs": {
      "vae_name": "taesd"
    },
    "class_type": "VAELoader"
  },
  "13": {
    "_meta": {
      "title": "TorchCompileLoadVAE"
    },
    "inputs": {
      "vae": [
        "11",
        0
      ],
      "mode": "reduce-overhead",
      "backend": "inductor",
      "fullgraph": true,
      "compile_decoder": true,
      "compile_encoder": true
    },
    "class_type": "TorchCompileLoadVAE"
  },
  "14": {
    "_meta": {
      "title": "VAE Decode"
    },
    "inputs": {
      "vae": [
        "13",
        0
      ],
      "samples": [
        "7",
        0
      ]
    },
    "class_type": "VAEDecode"
  },
  "15": {
    "_meta": {
      "title": "Preview Image"
    },
    "inputs": {
      "images": [
        "14",
        0
      ]
    },
    "class_type": "PreviewImage"
  },
  "16": {
    "_meta": {
      "title": "Empty Latent Image"
    },
    "inputs": {
      "width": 512,
      "height": 512,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage"
  },
  "23": {
    "_meta": {
      "title": "Load CLIP"
    },
    "inputs": {
      "type": "stable_diffusion",
      "device": "default",
      "clip_name": "CLIPText/model.fp16.safetensors"
    },
    "class_type": "CLIPLoader"
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
    def __init__(self, **params):
        super().__init__(**params)

        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams

        self.update_params(**params)

        # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
        # We can run the prompt once before actual inputs come in to "warmup"
        warmup_input = torch.randn(1, 512, 512, 3)
        asyncio.get_event_loop().run_until_complete(self.client.queue_prompt(warmup_input))

    def process_frame(self, image: Image.Image) -> Image.Image:
        # Normalize by dividing by 255 to ensure the tensor values are between 0 and 1
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        # Convert from numpy to torch.Tensor
        # Initially, the torch.Tensor will have shape HWC but we want BHWC
        # unsqueeze(0) will add a batch dimension at the beginning of 1 which means we just have 1 image
        image_tensor = torch.tensor(image_np).unsqueeze(0)

        # Process using ComfyUI pipeline
        result_tensor = asyncio.get_event_loop().run_until_complete(self.client.queue_prompt(image_tensor))

        # Convert back from Tensor to PIL.Image
        result_tensor = result_tensor.squeeze(0)
        result_image_np = (result_tensor * 255).byte()
        result_image = Image.fromarray(result_image_np.cpu().numpy())
        return result_image

    def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"ComfyUI Pipeline Prompt: {new_params.prompt}")
        self.client.set_prompt(new_params.prompt)
        self.params = new_params

    #TODO: This is a hack to stop the ComfyStreamClient. Use the comfystream api to stop the client in 0.0.2
    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        if self.client.comfy_client.is_running:
            await self.client.comfy_client.__aexit__(None, None, None)
        logging.info("ComfyUI pipeline stopped")

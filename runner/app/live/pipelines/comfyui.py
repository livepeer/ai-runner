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
    "inputs": {
      "image": "DALL·E 2024-11-15 10.15.49 - An anime-style character standing in a modern office space. The character is a young professional, dressed in a stylish business outfit, with medium-l.jpg",
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
  "4": {
    "inputs": {
      "ckpt_name": "SD1.5/dreamshaper-8.safetensors"
    },
    "class_type": "CheckpointLoaderSimple",
    "_meta": {
      "title": "Load Checkpoint"
    }
  },
  "5": {
    "inputs": {
      "text": "the hulk",
      "clip": [
        "4",
        1
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
        "4",
        1
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "7": {
    "inputs": {
      "seed": 945236422600751,
      "steps": 3,
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

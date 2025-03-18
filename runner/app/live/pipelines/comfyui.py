import os
import json
import torch
import numpy as np
import asyncio
import av
from PIL import Image
from typing import Union, Dict, List, Any
from pydantic import BaseModel, field_validator

from .interface import Pipeline
from comfystream.client import ComfyStreamClient

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
    def __init__(self, **params):
        super().__init__(**params)

        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace, **params)
        self.params: ComfyUIParams
        
        self.video_incoming_frames = asyncio.Queue()
        self.audio_incoming_frames = asyncio.Queue()
        self.processed_audio_buffer = np.array([], dtype=np.int16)

        # Use asyncio.run to handle update_params since it's now async
        asyncio.run(self.update_params(**params))

    async def warm_video(self):
        # Comfy will cache nodes that only need to be run once (i.e. a node that loads model weights)
        # We can run the prompt once before actual inputs come in to "warmup"
        dummy_frame = av.VideoFrame()
        dummy_frame.side_data.input = torch.randn(1, 512, 512, 3)

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            await self.client.get_video_output()
    
    async def warm_audio(self):
        dummy_frame = av.AudioFrame()
        dummy_frame.side_data.input = np.random.randint(-32768, 32767, int(48000 * 0.5), dtype=np.int16)
        dummy_frame.sample_rate = 48000

        for _ in range(WARMUP_RUNS):
            self.client.put_audio_input(dummy_frame)
            await self.client.get_audio_output()

    def video_preprocess(self, frame: av.VideoFrame) -> Union[torch.Tensor, np.ndarray]:
        frame_np = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return torch.from_numpy(frame_np).unsqueeze(0)
    
    def audio_preprocess(self, frame: av.AudioFrame) -> Union[torch.Tensor, np.ndarray]:
        return frame.to_ndarray().ravel().reshape(-1, 2).mean(axis=1).astype(np.int16)
    
    def video_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.VideoFrame:
        return av.VideoFrame.from_ndarray(
            (output * 255.0).clamp(0, 255).to(dtype=torch.uint8).squeeze(0).cpu().numpy()
        )

    def audio_postprocess(self, output: Union[torch.Tensor, np.ndarray]) -> av.AudioFrame:
        return av.AudioFrame.from_ndarray(np.repeat(output, 2).reshape(1, -1))

    async def put_video_frame(self, frame: av.VideoFrame):
        frame.side_data.input = self.video_preprocess(frame)
        frame.side_data.skipped = True
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(frame)

    async def put_audio_frame(self, frame: av.AudioFrame):
        frame.side_data.input = self.audio_preprocess(frame)
        frame.side_data.skipped = True
        self.client.put_audio_input(frame)
        await self.audio_incoming_frames.put(frame)

    async def get_processed_video_frame(self):
        out_tensor = await self.client.get_video_output()
        frame = await self.video_incoming_frames.get()
        while frame.side_data.skipped:
            frame = await self.video_incoming_frames.get()

        processed_frame = self.video_postprocess(out_tensor)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        
        return processed_frame

    async def get_processed_audio_frame(self):
        frame = await self.audio_incoming_frames.get()
        if frame.samples > len(self.processed_audio_buffer):
            out_tensor = await self.client.get_audio_output()
            self.processed_audio_buffer = np.concatenate([self.processed_audio_buffer, out_tensor])
        out_data = self.processed_audio_buffer[:frame.samples]
        self.processed_audio_buffer = self.processed_audio_buffer[frame.samples:]

        processed_frame = self.audio_postprocess(out_data)
        processed_frame.pts = frame.pts
        processed_frame.time_base = frame.time_base
        processed_frame.sample_rate = frame.sample_rate
        
        return processed_frame

    # For backward compatibility
    def process_frame(self, image: Image.Image) -> Image.Image:
        # Normalize by dividing by 255 to ensure the tensor values are between 0 and 1
        # image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        
        #image_np = self.video_preprocess(image_np)
        
        # Convert from numpy to torch.Tensor
        # Initially, the torch.Tensor will have shape HWC but we want BHWC
        # unsqueeze(0) will add a batch dimension at the beginning of 1 which means we just have 1 image
        #image_tensor = torch.tensor(image_np).unsqueeze(0)

        # Process using ComfyUI pipeline
        result_tensor = asyncio.get_event_loop().run_until_complete(self.put_video_frame(image))
        video_postprocess = self.video_postprocess(result_tensor)
#        result_tensor = asyncio.get_event_loop().run_until_complete(self.client.run_(image_tensor))

        result_image = self.video_postprocess(video_postprocess)

        # # Convert back from Tensor to PIL.Image
        # result_tensor = result_tensor.squeeze(0)
        # result_image_np = (result_tensor * 255).byte()
        # result_image = Image.fromarray(result_image_np.cpu().numpy())
        return result_image

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"ComfyUI Pipeline Prompt: {new_params.prompt}")
        await self.set_prompts(new_params.prompt)
        self.params = new_params

    async def set_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(prompts, list):
            await self.client.set_prompts(prompts)
        else:
            await self.client.set_prompts([prompts])

    async def update_prompts(self, prompts: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(prompts, list):
            await self.client.update_prompts(prompts)
        else:
            await self.client.update_prompts([prompts])
    
    async def get_nodes_info(self) -> Dict[str, Any]:
        """Get information about all nodes in the current prompt including metadata."""
        nodes_info = await self.client.get_available_nodes()
        return nodes_info

    # Modified stop method to use cleanup
    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.cleanup()
        logging.info("ComfyUI pipeline stopped")
    
    async def cleanup(self):
        await self.client.cleanup()

import os
import json
import torch
import asyncio
import numpy as np
from PIL import Image
from typing import Union
from pydantic import BaseModel, field_validator
import pathlib

from .interface import Pipeline
from comfystream.client import ComfyStreamClient
from trickle import VideoFrame, VideoOutput

import logging

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
WARMUP_RUNS = 1

_default_workflow_path = pathlib.Path(__file__).parent.absolute() / "comfyui_default_workflow.json"
with open(_default_workflow_path, 'r') as f:
    DEFAULT_WORKFLOW_JSON = json.load(f)


class ComfyUIParams(BaseModel):
    class Config:
        extra = "forbid"

    prompt: Union[str, dict] = DEFAULT_WORKFLOW_JSON
    width: int = 512
    height: int = 512

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


class ComfyUtils:
    @staticmethod
    def get_latent_image_dimensions(workflow: dict) -> tuple[int, int]:
        """Get dimensions from the EmptyLatentImage node in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary
            
        Returns:
            Tuple of (width, height) from the latent image, or (None, None) if not found
        """
        for node_id, node in workflow.items():
            if node.get("class_type") == "EmptyLatentImage":
                try:
                    inputs = node.get("inputs", {})
                    return inputs.get("width"), inputs.get("height")
                except Exception as e:
                    logging.warning(f"Failed to extract dimensions from latent image: {e}")
                    return None, None
        return None, None

    @staticmethod
    def update_latent_image_dimensions(workflow: dict, width: int, height: int) -> dict | None:
        """Update the EmptyLatentImage node dimensions in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary
            width: Width to set
            height: Height to set
        """
        for node_id, node in workflow.items():
            if node.get("class_type") == "EmptyLatentImage":
                try:
                    if "inputs" not in node:
                        node["inputs"] = {}
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height
                    logging.info(f"Updated latent image dimensions to {width}x{height}")
                except Exception as e:
                    logging.warning(f"Failed to update latent image dimensions: {e}")
                break


class ComfyUI(Pipeline):
    def __init__(self):
        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()

    async def initialize(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        
        # Get dimensions from workflow if it's a dict
        width = params.get('width')
        height = params.get('height')
        
        if width is None or height is None:
            if isinstance(new_params.prompt, dict):
                # If dimensions not provided in params, get them from latent image
                latent_width, latent_height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
                new_params.width = width or latent_width or new_params.width
                new_params.height = height or latent_height or new_params.height
            else:
                # If dimensions provided in params, update the latent image
                ComfyUtils.update_latent_image_dimensions(new_params.prompt, width, height)

        # TODO clean up extra vars
        width = width or new_params.width
        height = height or new_params.height
    
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params

        # Warm up the pipeline with the final dimensions
        logging.info(f"Warming up pipeline with dimensions: {width}x{height}")
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.input = torch.randn(1, height, width, 3)

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            _ = await self.client.get_video_output()
        logging.info("Pipeline initialization and warmup complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        image_np = np.array(frame.image.convert("RGB")).astype(np.float32) / 255.0
        frame.side_data.input = torch.tensor(image_np).unsqueeze(0)
        frame.side_data.skipped = True
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(VideoOutput(frame, request_id))

    async def get_processed_video_frame(self):
        result_tensor = await self.client.get_video_output()
        out = await self.video_incoming_frames.get()
        while out.frame.side_data.skipped:
            out = await self.video_incoming_frames.get()

        result_tensor = result_tensor.squeeze(0)
        result_image_np = (result_tensor * 255).byte()
        result_image = Image.fromarray(result_image_np.cpu().numpy())
        return out.replace_image(result_image)

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.update_prompts([new_params.prompt])
        self.params = new_params

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.client.cleanup()
        logging.info("ComfyUI pipeline stopped")

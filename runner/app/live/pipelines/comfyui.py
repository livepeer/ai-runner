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


class ComfyUI(Pipeline):
    def __init__(self):
        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()

    async def initialize(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params

        # Use warm_video with params dimensions
        await self.warm_video(self.params.width, self.params.height, WARMUP_RUNS)

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
        if new_params.width is not None and new_params.height is not None:
            logging.info(f"Updating resolution to {new_params.width}x{new_params.height}")
            self.params.width = new_params.width
            self.params.height = new_params.height
            await self.client.update_resolution(new_params.width, new_params.height, WARMUP_RUNS)
        else:
            logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
            await self.client.update_prompts([new_params.prompt])
        self.params = new_params

    def get_pipeline_width(self) -> int:
        return self.params.width if self.params and self.params.width is not None else 512
        
    def get_pipeline_height(self) -> int:
        return self.params.height if self.params and self.params.height is not None else 512
        
    async def warm_video(self, width: int, height: int, num_runs: int = 5):
        """Warm up the video processing pipeline with dummy frames."""
        await self.client.warm_video(width, height, num_runs)

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.client.cleanup()
        logging.info("ComfyUI pipeline stopped")

    async def update_resolution(self, width: int, height: int):
        """Update pipeline resolution.
        
        Implementation of abstract method from Pipeline interface.
        Delegates to update_params to maintain single source of truth for parameter updates.
        """
        await self.update_params(width=width, height=height)

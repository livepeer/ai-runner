import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Union
from pydantic import BaseModel, field_validator
import pathlib

from .interface import Pipeline
from comfystream.pipeline import Pipeline as ComfyStreamPipeline
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
        self.pipeline = ComfyStreamPipeline(width=512, height=512, cwd=comfy_ui_workspace)
        self.params: ComfyUIParams

    async def initialize(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        await self.pipeline.set_prompts([new_params.prompt])
        self.params = new_params

        # Warm up the pipeline
        await self.pipeline.warm_video()
        logging.info("Pipeline initialization and warmup complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        # Convert VideoFrame to format expected by comfystream
        image_np = np.array(frame.image.convert("RGB")).astype(np.float32) / 255.0
        frame.side_data.input = torch.tensor(image_np).unsqueeze(0)
        frame.side_data.skipped = True
        frame.side_data.request_id = request_id
        await self.pipeline.put_video_frame(frame)

    async def get_processed_video_frame(self) -> VideoOutput:
        processed_frame = await self.pipeline.get_processed_video_frame()
        # Convert back to VideoOutput format
        result_tensor = processed_frame.side_data.input
        result_tensor = result_tensor.squeeze(0)
        result_image_np = (result_tensor * 255).byte()
        result_image = Image.fromarray(result_image_np.cpu().numpy())
        return VideoOutput(processed_frame, processed_frame.side_data.request_id).replace_image(result_image)

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        await self.pipeline.update_prompts([new_params.prompt])
        self.params = new_params

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.pipeline.cleanup()
        logging.info("ComfyUI pipeline stopped")

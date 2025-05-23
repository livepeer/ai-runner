import os
import json
import PIL
import torch
import asyncio
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

        # Get dimensions from the workflow
        if isinstance(new_params.prompt, dict):
            width, height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
            if width is None or height is None:
                width, height = ComfyUtils.DEFAULT_WIDTH, ComfyUtils.DEFAULT_HEIGHT  # Default dimensions if not found in workflow
                logging.warning(f"Could not find dimensions in workflow, using default {width}x{height}")
    
        width, height = width or ComfyUtils.DEFAULT_WIDTH, height or ComfyUtils.DEFAULT_HEIGHT
        
        # Warm up the pipeline with the workflow dimensions
        logging.info(f"Warming up pipeline with dimensions: {width}x{height}")
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.input = torch.randn(1, height, width, 3)

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            _ = await self.client.get_video_output()
        logging.info("Pipeline initialization and warmup complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        tensor = frame.tensor
        if tensor.is_cuda:
            # Clone the tensor to be able to send it on comfystream internal queue
            tensor = tensor.clone()
        frame.side_data.input = tensor
        frame.side_data.skipped = True
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(VideoOutput(frame, request_id))

    async def get_processed_video_frame(self):
        result_tensor = await self.client.get_video_output()
        out = await self.video_incoming_frames.get()
        while out.frame.side_data.skipped:
            out = await self.video_incoming_frames.get()
        return out.replace_tensor(result_tensor)

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        try:
            await self.client.update_prompts([new_params.prompt])
        except Exception as e:
            logging.error(f"Error updating ComfyUI Pipeline Prompt: {e}")
            raise e
        self.params = new_params

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.client.cleanup()
        logging.info("ComfyUI pipeline stopped")

class ComfyUtils:
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    
    @staticmethod
    def get_latent_image_dimensions(workflow: dict) -> tuple[int, int]:
        """Get dimensions from the EmptyLatentImage node in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary
            
        Returns:
            Tuple of (width, height) from the latent image. Returns default dimensions if not found or on error.
        """
        try:
            for node_id, node in workflow.items():
                if node.get("class_type") == "EmptyLatentImage":
                    inputs = node.get("inputs", {})
                    width = inputs.get("width")
                    height = inputs.get("height")
                    if width is not None and height is not None:
                        return width, height
                    logging.warning("Incomplete dimensions in latent image node")
                    break
        except Exception as e:
            logging.warning(f"Failed to extract dimensions from workflow: {e}")
        
        # Return defaults if dimensions not found or on any error
        logging.info(f"Using default dimensions {ComfyUtils.DEFAULT_WIDTH}x{ComfyUtils.DEFAULT_HEIGHT}")
        return ComfyUtils.DEFAULT_WIDTH, ComfyUtils.DEFAULT_HEIGHT

import os
import json
import torch
import asyncio
from typing import Union
from pydantic import BaseModel, field_validator
import pathlib

from .interface import Pipeline
from comfystream.client import ComfyStreamClient
from trickle import VideoFrame, VideoOutput, DEFAULT_WIDTH, DEFAULT_HEIGHT
from utils import ComfyUtils

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
        self.params: ComfyUIParams
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.pause_input = False
        self.client = None

    async def initialize(self, **params):
        """Initialize the ComfyUI pipeline with given parameters."""
        self.client = ComfyStreamClient(cwd=os.getenv(COMFY_UI_WORKSPACE_ENV))
        new_params = ComfyUIParams(**params)
        self.width = new_params.width
        self.height = new_params.height
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        
        new_params.prompt = ComfyUtils.update_latent_image_dimensions(new_params.prompt, self.width, self.height)
        logging.info(f"Updated prompt with latent image dimensions {self.width}x{self.height} from request")

        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params

        # Warm up the pipeline
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.input = torch.randn(1, self.height, self.width, 3)

        for _ in range(WARMUP_RUNS):
            self.client.put_video_input(dummy_frame)
            _ = await self.client.get_video_output()
        logging.info("Pipeline initialization and warmup complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        if self.pause_input:
            logging.warning("ComfyUI pipeline is paused, skipping input frame")
            return

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
        try:
            logging.info("Stopping ComfyUI pipeline")
            self.pause_input = True
            
            # Wait for the pipeline to stop
            # Clear the video_incoming_frames queue
            while not self.video_incoming_frames.empty():
                try:
                    frame = self.video_incoming_frames.get_nowait()
                    # Ensure any CUDA tensors are properly handled
                    if frame.tensor is not None and frame.tensor.is_cuda:
                        frame.tensor.cpu()
                except asyncio.QueueEmpty:
                    break

            logging.info("Waiting for ComfyUI client to cleanup")
            await self.client.cleanup(exit_client=True, unload_models=True)
            await asyncio.sleep(1)
            logging.info("ComfyUI client cleanup complete")
                        
            # Force CUDA cache clear
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error stopping ComfyUI pipeline: {e}")
        finally:
            self.pause_input = False
            self.client = None
        self.video_incoming_frames = asyncio.Queue()
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT

        logging.info("ComfyUI pipeline stopped")
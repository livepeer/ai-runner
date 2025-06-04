import os
import json
import torch
import asyncio
from typing import Union
from pydantic import BaseModel, field_validator
import pathlib

from .interface import Pipeline
from comfystream.client import ComfyStreamClient
from trickle import VideoFrame, VideoOutput
from utils import ComfyUtils

import logging

COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
WARMUP_RUNS = 1

def get_default_workflow_json():
    _default_workflow_path = pathlib.Path(__file__).parent.absolute() / "comfyui_default_workflow.json"
    with open(_default_workflow_path, 'r') as f:
        return json.load(f)

# Get the default workflow json during startup
DEFAULT_WORKFLOW_JSON = get_default_workflow_json()

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
        self.width = ComfyUtils.DEFAULT_WIDTH
        self.height = ComfyUtils.DEFAULT_HEIGHT

    async def initialize(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        
        # Store the parameters
        self.params = new_params
        
        # Set the prompts in the client
        try:
            await self.client.set_prompts([new_params.prompt])
            
            # Get dimensions from the workflow to warm the pipeline
            width, height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
            self.width = width
            self.height = height
            
            # Warm up the pipeline with the workflow dimensions
            logging.info(f"Warming up pipeline with dimensions: {width}x{height}")
            dummy_frame = VideoFrame(None, 0, 0)
            dummy_frame.side_data.input = torch.randn(1, height, width, 3)

            for _ in range(WARMUP_RUNS):
                self.client.put_video_input(dummy_frame)
                _ = await self.client.get_video_output()
            logging.info("Pipeline initialization and warmup complete")
        except Exception as e:
            logging.error(f"Error initializing ComfyUI Pipeline: {e}")
            raise e

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
        
        try:
            # Attempt to get dimensions from the workflow
            width, height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
            
            # Check if dimensions have changed
            if width != self.width or height != self.height:
                logging.info(f"Pipeline dimensions updated: {self.width}x{self.height} -> {width}x{height}")
                
                # Stop the pipeline quickly
                logging.info("Stopping pipeline for dimension change")
                #await self.client.cleanup()
                
                # Update dimensions
                self.width = width
                self.height = height
                
                # Clear all queues
                self.video_incoming_frames.empty()
                
                # Reinitialize with new dimensions
                logging.info(f"Reinitializing pipeline with new dimensions: {width}x{height}")
                await self.client.set_prompts([new_params.prompt])
                
                # Warm up with new dimensions
                dummy_frame = VideoFrame(None, 0, 0)
                dummy_frame.side_data.input = torch.randn(1, height, width, 3)
                
                for _ in range(WARMUP_RUNS):
                    self.client.put_video_input(dummy_frame)
                    _ = await self.client.get_video_output()
                
                logging.info("Pipeline reinitialized with new dimensions")
            else:
                # Only update prompts if dimensions haven't changed
                logging.info("Dimensions unchanged, updating prompts only")
                await self.client.set_prompts([new_params.prompt])
                await self.client.update_prompts([new_params.prompt])
            
            self.params = new_params
            logging.info("Pipeline parameters updated successfully")
        except Exception as e:
            logging.error(f"Error updating ComfyUI Pipeline parameters: {e}")
            # Restore previous state on error
            try:
                await self.client.set_prompts([self.params.prompt])
                await self.client.update_prompts([self.params.prompt])
            except Exception as restore_error:
                logging.error(f"Failed to restore previous state: {restore_error}")
            raise e

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        try:
            # Clear the video incoming frames queue and move any CUDA tensors to CPU
            while not self.video_incoming_frames.empty():
                try:
                    frame = self.video_incoming_frames.get_nowait()
                    if frame.tensor.is_cuda:
                        frame.tensor.cpu()  # Move tensor to CPU before deletion
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logging.error(f"Error clearing video incoming frames queue: {e}")

            # Now cleanup the client with a timeout
            try:
                async with asyncio.timeout(5.0):  # 5 second timeout for client cleanup
                    await self.client.cleanup()
            except asyncio.TimeoutError:
                logging.error("Timeout during client cleanup")
            except Exception as e:
                logging.error(f"Error during client cleanup: {e}")

            # Force CUDA cache clear
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        except Exception as e:
            logging.error(f"Error during ComfyUI pipeline cleanup: {e}")
        finally:
            logging.info("ComfyUI pipeline stopped")
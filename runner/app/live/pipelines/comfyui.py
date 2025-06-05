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
        self.comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=self.comfy_ui_workspace)
        self.params: ComfyUIParams
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.width = ComfyUtils.DEFAULT_WIDTH
        self.height = ComfyUtils.DEFAULT_HEIGHT
        self.pause_frames = False

    async def initialize(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params
        
        # Get dimensions from the workflow to warm the pipeline
        width, height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
        
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
        if self.pause_frames or self.client.is_shutting_down:
            # Skip frames if pipeline is paused or shutting down
            return
        self.client.put_video_input(frame)
        await self.video_incoming_frames.put(VideoOutput(frame, request_id))            
            
    async def get_processed_video_frame(self):
        result_tensor = await self.client.get_video_output()
        out = await self.video_incoming_frames.get()
        while out.frame.side_data.skipped or self.pause_frames:
            if self.client.is_shutting_down:
                raise Exception("Client is shutting down, skipping frame")
            else:
                out = await self.video_incoming_frames.get()
        return out.replace_tensor(result_tensor)

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        
        try:
            width, height = ComfyUtils.get_latent_image_dimensions(new_params.prompt)
            if width != self.width or height != self.height:
                self.width = width
                self.height = height
                logging.info(f"pipeline dimensions updated, clearing queues: {self.width}x{self.height} -> {width}x{height}")
                self.pause_frames = True
                if self.client.comfy_client.is_running:
                    logging.info("comfystream is running, exiting")
                    await self.client.comfy_client.__aexit__()
                    self.client = ComfyStreamClient(cwd=self.comfy_ui_workspace)
                    logging.info(f"comfystream rescyled: {self.width}x{self.height}")
                    await self.client.set_prompts([new_params.prompt])
                    await self.client.update_prompts([new_params.prompt])
                self.pause_frames = False
            else:
                logging.info(f"pipeline dimensions unchanged: {self.width}x{self.height}")
                await self.client.set_prompts([new_params.prompt])
                await self.client.update_prompts([new_params.prompt])
            
            logging.info("pipeline dimensions updated, re-initialized")
            
        except Exception as e:
            logging.error(f"Error updating ComfyUI Pipeline Prompt: {e}")
            raise e
        self.params = new_params

    async def stop(self):
        """Stop the ComfyUI pipeline and ensure all resources are cleaned up"""
        logging.info("Stopping ComfyUI pipeline")
        self.pause_frames = True
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
            if self.client is not None:
                try:
                    # Then do full cleanup with timeout
                    async with asyncio.timeout(10.0):  # Increased timeout for client cleanup
                        await self.client.cleanup()
                except asyncio.TimeoutError:
                    logging.error("Timeout during client cleanup")
                except Exception as e:
                    logging.error(f"Error during client cleanup: {e}")
                finally:
                    self.client = None

            # Final CUDA cleanup after all operations are complete
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                except Exception as e:
                    logging.error(f"Error during final CUDA cleanup: {e}")

        except Exception as e:
            logging.error(f"Error during ComfyUI pipeline cleanup: {e}")
            raise
        finally:
            logging.info("ComfyUI pipeline stopped")
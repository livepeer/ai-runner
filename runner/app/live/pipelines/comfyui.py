import os
import torch
import asyncio
import logging

from comfystream.client import ComfyStreamClient

from .interface import Pipeline
from .comfyui_params import ComfyUIParams
from ..trickle import VideoFrame, VideoOutput


COMFY_UI_WORKSPACE_ENV = "COMFY_UI_WORKSPACE"
WARMUP_RUNS = 1


class ComfyUI(Pipeline):
    def __init__(self):
        comfy_ui_workspace = os.getenv(COMFY_UI_WORKSPACE_ENV)
        self.client = ComfyStreamClient(cwd=comfy_ui_workspace)
        self.params: ComfyUIParams
        self.video_incoming_frames: asyncio.Queue[VideoOutput] = asyncio.Queue()

    async def initialize(self, **params):
        """Initialize the ComfyUI pipeline with given parameters."""
        new_params = ComfyUIParams(**params)
        logging.info(f"Initializing ComfyUI Pipeline with prompt: {new_params.prompt}")
        # TODO: currently its a single prompt, but need to support multiple prompts
        await self.client.set_prompts([new_params.prompt])
        self.params = new_params

        # Warm up the pipeline
        dummy_frame = VideoFrame(None, 0, 0)
        dummy_frame.side_data.input = torch.randn(
            1, new_params.height, new_params.width, 3
        )

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
        try:
            logging.info("Stopping ComfyUI pipeline")
            # Wait for the pipeline to stop
            # Clear the video_incoming_frames queue
            while not self.video_incoming_frames.empty():
                try:
                    self.video_incoming_frames.get_nowait()
                except asyncio.QueueEmpty:
                    break
            logging.info("Waiting for ComfyUI client to cleanup")
            await self.client.cleanup()
            await asyncio.sleep(1)
            logging.info("ComfyUI client cleanup complete")
            # Force CUDA cache clear
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error stopping ComfyUI pipeline: {e}")
        finally:
            self.client = None
            logging.info("ComfyUI pipeline stopped")

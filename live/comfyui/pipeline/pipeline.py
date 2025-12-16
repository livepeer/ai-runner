import asyncio
import logging
import os
from pathlib import Path

import torch
from comfystream.client import ComfyStreamClient

from app.live.pipelines import Pipeline
from app.live.trickle import VideoFrame, VideoOutput
from .params import ComfyUIParams


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
        frame.side_data.input = frame.tensor
        frame.side_data.skipped = True
        out_frame = VideoOutput(frame.replace_tensor(torch.zeros_like(frame.tensor)), request_id)
        await self.video_incoming_frames.put(out_frame)
        self.client.put_video_input(frame)

    async def get_processed_video_frame(self):
        result_tensor = await self.client.get_video_output()
        out = await self.video_incoming_frames.get()
        while out.frame.side_data.skipped:
            out = await self.video_incoming_frames.get()
        return out.replace_tensor(result_tensor)

    async def update_params(self, **params):
        update_task = asyncio.create_task(self._do_update_params(**params))

        try:
            await asyncio.wait_for(asyncio.shield(update_task), timeout=2.0)
        except asyncio.TimeoutError:
            logging.info("Update taking a while, returning task for loading overlay")
            return update_task

    async def _do_update_params(self, **params):
        """Perform the actual parameter update with logging and param setting."""
        new_params = ComfyUIParams(**params)
        if new_params == self.params:
            logging.info("No parameters changed")
            return

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

    @classmethod
    def prepare_models(cls):
        raise NotImplementedError(
            "ComfyUI uses a separate model preparation flow. "
            "See dl_checkpoints.sh download_comfyui_live_models()."
        )

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Union
from pydantic import BaseModel, field_validator
import pathlib
import av

from .interface import Pipeline
from comfystream.pipeline import Pipeline as ComfyStreamPipeline
from trickle import VideoFrame, VideoOutput, AudioFrame, AudioOutput

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
        await self.pipeline.put_video_frame(self._convert_to_av_frame(frame))

    async def put_audio_frame(self, frame: AudioFrame, request_id: str):
        await self.pipeline.put_audio_frame(self._convert_to_av_frame(frame))

    async def get_processed_video_frame(self, request_id: str) -> VideoOutput:
        av_frame = await self.pipeline.get_processed_video_frame()
        video_frame = VideoFrame.from_av_video(av_frame)
        video_frame.side_data.request_id = request_id
        return VideoOutput(video_frame).replace_image(av_frame.to_image())

    async def get_processed_audio_frame(self, request_id: str) -> AudioOutput:        
        av_frame = await self.pipeline.get_processed_audio_frame()
        return AudioOutput(av_frame, request_id)

    async def update_params(self, **params):
        new_params = ComfyUIParams(**params)
        logging.info(f"Updating ComfyUI Pipeline Prompt: {new_params.prompt}")
        await self.pipeline.update_prompts([new_params.prompt])
        self.params = new_params

    async def stop(self):
        logging.info("Stopping ComfyUI pipeline")
        await self.pipeline.cleanup()
        logging.info("ComfyUI pipeline stopped")

    def _convert_to_av_frame(self, frame: Union[VideoFrame, AudioFrame]) -> Union[av.VideoFrame, av.AudioFrame]:
        """Convert trickle frame to av frame"""
        if isinstance(frame, VideoFrame):
            av_frame = av.VideoFrame.from_ndarray(
                np.array(frame.image.convert("RGB")), 
                format='rgb24'
            )
        elif isinstance(frame, AudioFrame):
            av_frame = av.AudioFrame.from_ndarray(
                frame.samples.reshape(-1, 1),
                layout='mono',
                rate=frame.rate
            )
        
        # Common frame properties
        av_frame.pts = frame.timestamp
        av_frame.time_base = frame.time_base
        return av_frame

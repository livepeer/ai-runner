import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from scope.core.pipelines.interface import Pipeline as ScopePipeline

from runner.live.pipelines import Pipeline
from runner.live.trickle import VideoFrame, VideoOutput
from .params import ScopeParams

# Models directory configured via DAYDREAM_SCOPE_MODELS_DIR env var
MODELS_DIR = Path(os.environ.get("DAYDREAM_SCOPE_MODELS_DIR", "/models/Scope--models"))


class Scope(Pipeline):
    def __init__(self):
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self.pipe: Optional[ScopePipeline] = None
        self.params: Optional[ScopeParams] = None

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        """
        Generate frames on each input frame trigger.

        Input frame content is ignored - we use it as a trigger for continuous generation.
        Each call generates a batch of frames which are all queued individually.
        """
        if self.pipe is None:
            logging.warning("Pipeline not initialized, dropping frame")
            return

        # Measure generation time to calculate FPS
        start_time = time.time()
        output = await asyncio.to_thread(self._generate_sync)
        generation_time = time.time() - start_time

        # Queue each generated frame individually
        # output shape is (T, H, W, C) where T is number of frames
        num_frames = output.shape[0]

        # Calculate time per frame (in seconds)
        time_per_frame = generation_time / num_frames if num_frames > 0 else 1.0 / 16.0

        # Convert time per frame to timestamp increment in time_base units
        # timestamp_increment = time_per_frame / time_base
        timestamp_increment = int(time_per_frame * frame.time_base.denominator / frame.time_base.numerator)
        current_timestamp = frame.timestamp

        for i in range(num_frames):
            # Extract single frame and add batch dimension: (H, W, C) -> (1, H, W, C)
            frame_tensor = output[i : i + 1]

            # Create new frame with incremented timestamp
            new_frame = VideoFrame(
                frame_tensor,
                current_timestamp,
                frame.time_base,
                frame.log_timestamps.copy(),
            )
            new_frame.side_data = frame.side_data

            video_output = VideoOutput(new_frame, request_id)
            await self.frame_queue.put(video_output)

            # Increment timestamp for next frame
            current_timestamp += timestamp_increment

    def _generate_sync(self) -> torch.Tensor:
        """Synchronous generation using the longlive pipeline."""
        assert self.pipe is not None
        assert self.params is not None

        # Convert params to scope format
        prompts = [
            {"text": p, "weight": 100} if isinstance(p, str)
            else {"text": p.text, "weight": p.weight}
            for p in self.params.prompts
        ]

        # Generate frames - returns (T, H, W, C) tensor
        output = self.pipe(prompts=prompts)
        return output

    async def get_processed_video_frame(self) -> VideoOutput:
        return await self.frame_queue.get()

    async def initialize(self, **params):
        logging.info(f"Initializing Scope pipeline with params: {params}")

        self.params = ScopeParams(**params)

        # Load the pipeline based on the pipeline type
        if self.params.pipeline == "longlive":
            self.pipe = await asyncio.to_thread(_load_longlive_pipeline, self.params)
        else:
            raise ValueError(f"Unsupported pipeline: {self.params.pipeline}")

        logging.info("Pipeline initialization complete")


    async def update_params(self, **params):
        """Update pipeline parameters."""
        logging.info(f"Updating params: {params}")
        new_params = ScopeParams(**params)

        # Check if we need to reload the pipeline
        needs_reload = (
            self.params is None
            or new_params.pipeline != self.params.pipeline
            or new_params.width != self.params.width
            or new_params.height != self.params.height
        )

        if needs_reload:
            logging.info("Parameters require pipeline reload")
            return asyncio.create_task(self._reload_pipeline(new_params))

        # Update params without reloading
        self.params = new_params
        return None

    async def _reload_pipeline(self, new_params: ScopeParams):
        """Reload the pipeline with new parameters."""
        self.pipe = None
        self.params = new_params

        if new_params.pipeline == "longlive":
            self.pipe = await asyncio.to_thread(_load_longlive_pipeline, new_params)
        else:
            raise ValueError(f"Unsupported pipeline: {new_params.pipeline}")

    async def stop(self):
        logging.info("Stopping pipeline")
        self.pipe = None
        self.params = None
        self.frame_queue = asyncio.Queue()

    @classmethod
    def prepare_models(cls):
        """Download all scope models."""
        logging.info("Preparing Scope models")
        logging.info(f"Models directory: {MODELS_DIR}")

        # Import and call scope's download function directly
        from scope.server.download_models import download_models

        download_models()  # Downloads all scope pipelines

        logging.info("Scope model preparation complete")


def _load_longlive_pipeline(params: ScopeParams) -> ScopePipeline:
    """Load the LongLive pipeline synchronously.

    Args:
        params: ScopeParams instance with pipeline configuration

    Returns:
        Pipeline instance from scope.core.pipelines
    """
    from scope.core.pipelines import LongLivePipeline

    logging.info(f"Loading LongLive pipeline from {MODELS_DIR}")

    config = OmegaConf.create(
        {
            "model_dir": str(MODELS_DIR),
            "generator_path": str(
                MODELS_DIR / "LongLive-1.3B" / "models" / "longlive_base.pt"
            ),
            "lora_path": str(MODELS_DIR / "LongLive-1.3B" / "models" / "lora.pt"),
            "text_encoder_path": str(
                MODELS_DIR / "WanVideo_comfy" / "umt5-xxl-enc-fp8_e4m3fn.safetensors"
            ),
            "tokenizer_path": str(
                MODELS_DIR / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"
            ),
            "height": params.height,
            "width": params.width,
            "seed": params.seed,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
    logging.info("LongLive pipeline loaded successfully")
    return pipe

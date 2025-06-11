import logging
import asyncio
from typing import Dict, List, Literal, Optional

import torch
from pydantic import BaseModel
from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput


class StreamDiffusionParams(BaseModel):
    class Config:
        extra = "forbid"

    prompt: str = "talking head, cyberpunk, tron, matrix, ultra-realistic, dark, futuristic, neon, 8k"
    model_id: str = "KBlueLeaf/kohaku-v2.1"
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    num_inference_steps: int = 50
    t_index_list: Optional[List[int]] = [37, 45, 48]
    scale: float = 1.0
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    use_denoising_batch: bool = True
    enable_similar_image_filter: bool = False
    seed: int = 2
    guidance_scale: float = 1.2
    do_add_noise: bool = False
    similar_image_filter_threshold: float = 0.98


class StreamDiffusion(Pipeline):
    def __init__(self):
        super().__init__()
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.first_frame = True
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()

    async def initialize(self, **params):
        logging.info(f"Initializing StreamDiffusion pipeline with params: {params}")
        await self.update_params(**params)
        logging.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        out_tensor = await asyncio.to_thread(self.process_tensor_sync, frame.tensor)
        output = VideoOutput(frame, request_id).replace_tensor(out_tensor)
        await self.frame_queue.put(output)

    def process_tensor_sync(self, img_tensor: torch.Tensor):
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        # The incoming frame.tensor is (B, H, W, C) in range [-1, 1] while the
        # VaeImageProcessor inside the wrapper expects (B, C, H, W) in [0, 1].
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = self.pipe.stream.image_processor.denormalize(img_tensor)
        img_tensor = self.pipe.preprocess_image(img_tensor)

        if self.first_frame:
            self.first_frame = False
            for _ in range(self.pipe.batch_size):
                self.pipe(image=img_tensor)

        out_tensor = self.pipe(image=img_tensor)
        if isinstance(out_tensor, list):
            out_tensor = out_tensor[0]

        # The output tensor from the wrapper is (C, H, W), and the encoder expects (1, H, W, C).
        return out_tensor.permute(1, 2, 0).unsqueeze(0)

    async def get_processed_video_frame(self) -> VideoOutput:
        return await self.frame_queue.get()

    async def update_params(self, **params):
        new_params = StreamDiffusionParams(**params)
        if self.pipe is not None:
            # avoid resetting the pipe if only the prompt changed
            only_prompt = self.params.model_copy(update={"prompt": new_params.prompt})
            if new_params == only_prompt:
                logging.info(f"Updating prompt: {new_params.prompt}")
                self.pipe.stream.update_prompt(new_params.prompt)
                self.params = new_params
                return

        logging.info(f"Resetting diffuser for params change")

        self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
        self.params = new_params
        self.first_frame = True

    async def stop(self):
        logging.info("Stopping StreamDiffusion pipeline")
        self.pipe = None


def load_streamdiffusion_sync(params: StreamDiffusionParams):
    pipe = StreamDiffusionWrapper(
        output_type="pt",
        model_id_or_path=params.model_id,
        lora_dict=params.lora_dict,
        use_lcm_lora=params.use_lcm_lora,
        lcm_lora_id=params.lcm_lora_id,
        t_index_list=params.t_index_list,
        frame_buffer_size=1,
        width=512,
        height=512,
        warmup=10,
        acceleration=params.acceleration,
        do_add_noise=params.do_add_noise,
        mode="img2img",
        enable_similar_image_filter=params.enable_similar_image_filter,
        similar_image_filter_threshold=params.similar_image_filter_threshold,
        use_denoising_batch=params.use_denoising_batch,
        seed=params.seed,
    )
    pipe.prepare(
        prompt=params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
    )
    return pipe

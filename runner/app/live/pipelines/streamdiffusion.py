import logging
import asyncio
import os
from typing import Dict, List, Literal, Optional

import torch
import numpy as np
from pydantic import BaseModel
from streamdiffusion import StreamDiffusionWrapper

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT


# Create debug logs directory
DEBUG_LOG_DIR = "/models/debug-logs"
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


def save_tensor_debug(tensor: torch.Tensor, name: str, step: str):
    """Save tensor as numpy array for debugging"""
    try:
        # Convert to numpy and save
        if tensor.dim() == 4:
            # Batch dimension, take first item
            tensor_np = tensor[0].detach().cpu().numpy()
        else:
            tensor_np = tensor.detach().cpu().numpy()

        # Normalize to 0-255 range for visualization
        if tensor_np.min() < 0 or tensor_np.max() > 1:
            # Assume range is [-1, 1], convert to [0, 1]
            tensor_np = (tensor_np + 1) / 2
        tensor_np = np.clip(tensor_np * 255, 0, 255).astype(np.uint8)

        # Save as numpy file
        filename = f"{DEBUG_LOG_DIR}/{step}_{name}.npy"
        np.save(filename, tensor_np)

        # Also save as image if it's 3-channel
        if tensor_np.shape[-1] == 3:
            try:
                import cv2
                # Convert from RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(tensor_np, cv2.COLOR_RGB2BGR)
                img_filename = f"{DEBUG_LOG_DIR}/{step}_{name}.png"
                cv2.imwrite(img_filename, img_bgr)
                logging.debug(f"Saved debug image: {img_filename}")
            except ImportError:
                logging.warning("OpenCV not available, skipping image save")

        logging.debug(f"Saved debug tensor {name} at step {step}: {filename}, shape: {tensor_np.shape}, range: [{tensor_np.min()}, {tensor_np.max()}]")

    except Exception as e:
        logging.error(f"Failed to save debug tensor {name}: {e}")


class StreamDiffusionParams(BaseModel):
    class Config:
        extra = "forbid"

    prompt: str = "talking head, cyberpunk, tron, matrix, ultra-realistic, dark, futuristic, neon, 8k"
    model_id: str = "KBlueLeaf/kohaku-v2.1"
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    num_inference_steps: int = 50
    t_index_list: List[int] = [37, 45, 48]
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
        self._pipeline_lock = asyncio.Lock()  # Protects pipeline initialization/reinitialization
        self.frame_count = 0

    async def initialize(self, **params):
        logging.info(f"Initializing StreamDiffusion pipeline with params: {params}")
        await self.update_params(**params)
        logging.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        async with self._pipeline_lock:
            self.frame_count += 1
            logging.debug(f"Processing frame {self.frame_count}, request_id: {request_id}")

            # Debug input tensor
            logging.debug(f"Input tensor shape: {frame.tensor.shape}, dtype: {frame.tensor.dtype}, range: [{frame.tensor.min():.3f}, {frame.tensor.max():.3f}]")
            save_tensor_debug(frame.tensor, "input", f"frame_{self.frame_count}")

            out_tensor = await asyncio.to_thread(self.process_tensor_sync, frame.tensor, self.frame_count)
            output = VideoOutput(frame, request_id).replace_tensor(out_tensor)
            await self.frame_queue.put(output)

    def process_tensor_sync(self, img_tensor: torch.Tensor, frame_num: int):
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        logging.debug(f"Processing tensor sync for frame {frame_num}")
        logging.debug(f"Initial tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}, range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

        # The incoming frame.tensor is (B, H, W, C) in range [-1, 1] while the
        # VaeImageProcessor inside the wrapper expects (B, C, H, W) in [0, 1].
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = self.pipe.stream.image_processor.denormalize(img_tensor)

        self.pipe.update_control_image_efficient(img_tensor)
        logging.debug("Updated control image")

        img_tensor = self.pipe.preprocess_image(img_tensor)
        logging.debug(f"After preprocess shape: {img_tensor.shape}, range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        save_tensor_debug(img_tensor, "after_preprocess", f"frame_{frame_num}")

        if self.first_frame:
            logging.info("Processing first frame - running warmup")
            self.first_frame = False
            for i in range(self.pipe.batch_size):
                logging.debug(f"Warmup iteration {i+1}/{self.pipe.batch_size}")
                warmup_output = self.pipe(image=img_tensor)
                if isinstance(warmup_output, list):
                    warmup_output = warmup_output[0]
                logging.debug(f"Warmup output shape: {warmup_output.shape}, range: [{warmup_output.min():.3f}, {warmup_output.max():.3f}]")
                save_tensor_debug(warmup_output, f"warmup_{i+1}", f"frame_{frame_num}")

        logging.debug("Running inference")
        out_tensor = self.pipe(image=img_tensor)
        logging.debug(f"Raw output type: {type(out_tensor)}, shape: {out_tensor.shape if hasattr(out_tensor, 'shape') else 'N/A'}")

        if isinstance(out_tensor, list):
            out_tensor = out_tensor[0]
            logging.debug(f"Extracted from list, shape: {out_tensor.shape}")

        logging.debug(f"Output tensor shape: {out_tensor.shape}, dtype: {out_tensor.dtype}, range: [{out_tensor.min():.3f}, {out_tensor.max():.3f}]")
        save_tensor_debug(out_tensor, "raw_output", f"frame_{frame_num}")

        # The output tensor format depends on the wrapper's output_type
        # For output_type="pt", it should be (C, H, W) in range [0, 1]
        # But let's handle a potential batch dimension and squeeze it.
        if out_tensor.dim() == 4:
            out_tensor = out_tensor.squeeze(0)
            logging.debug(f"After squeeze shape: {out_tensor.shape}")
        elif out_tensor.dim() != 3:
            raise ValueError(f"Unexpected output tensor dimensions: {out_tensor.shape}")

        # Now out_tensor should be (C, H, W), convert to (H, W, C)
        final_tensor = out_tensor.permute(1, 2, 0)
        logging.debug(f"Final tensor shape: {final_tensor.shape}, range: [{final_tensor.min():.3f}, {final_tensor.max():.3f}]")
        save_tensor_debug(final_tensor, "final_output", f"frame_{frame_num}")

        return final_tensor

    async def get_processed_video_frame(self) -> VideoOutput:
        return await self.frame_queue.get()

    async def update_params(self, **params):
        async with self._pipeline_lock:
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
            self.frame_count = 0

    async def stop(self):
        async with self._pipeline_lock:
            self.pipe = None
            self.frame_queue = asyncio.Queue()


def load_streamdiffusion_sync(params: StreamDiffusionParams):
    logging.info(f"Loading StreamDiffusion with model: {params.model_id}")
    logging.info(f"Model parameters: width={params.width}, height={params.height}, acceleration={params.acceleration}")
    logging.info(f"Inference parameters: steps={params.num_inference_steps}, guidance_scale={params.guidance_scale}")

    pipe = StreamDiffusionWrapper(
        output_type="pt",
        model_id_or_path=params.model_id,
        lora_dict=params.lora_dict,
        use_lcm_lora=params.use_lcm_lora,
        lcm_lora_id=params.lcm_lora_id,
        t_index_list=params.t_index_list,
        frame_buffer_size=1,
        width=params.width,
        height=params.height,
        warmup=10,
        acceleration=params.acceleration,
        do_add_noise=params.do_add_noise,
        mode="img2img",
        enable_similar_image_filter=params.enable_similar_image_filter,
        similar_image_filter_threshold=params.similar_image_filter_threshold,
        use_denoising_batch=params.use_denoising_batch,
        seed=params.seed,
        build_engines_if_missing=False,
    )

    logging.info("Preparing pipeline with prompt and parameters")
    pipe.prepare(
        prompt=params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
    )
    logging.info("StreamDiffusion pipeline loaded successfully")
    return pipe

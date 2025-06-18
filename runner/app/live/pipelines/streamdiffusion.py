import logging
import asyncio
from typing import Dict, List, Literal, Optional

import torch
from pydantic import BaseModel
from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT


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
        self._pipeline_lock = asyncio.Lock()  # Protects pipeline initialization/reinitialization

    async def initialize(self, **params):
        logging.info(f"Initializing StreamDiffusion pipeline with params: {params}")
        await self.update_params(**params)
        logging.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        async with self._pipeline_lock:
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

                # Check if resolution changed
                if new_params.width != self.params.width or new_params.height != self.params.height:
                    await self._change_resolution_locked(new_params)
                    return

            logging.info(f"Resetting diffuser for params change")

            self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
            self.params = new_params
            self.first_frame = True

    async def _change_resolution_locked(self, new_params: StreamDiffusionParams):
        """Internal method to change resolution. Must be called while holding the pipeline lock."""
        logging.info(f"Changing resolution from {self.params.width}x{self.params.height} to {new_params.width}x{new_params.height}")
        
        # Stop the current pipeline with timeout protection
        try:
            await asyncio.wait_for(self._stop_locked(), timeout=8.0)
        except asyncio.TimeoutError:
            logging.warning("Timeout during resolution change stop, forcing cleanup")
            self.pipe = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.warning(f"Error during forced CUDA cleanup: {e}")
        
        # Initialize with new parameters
        self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
        self.params = new_params
        self.first_frame = True
        
        logging.info("Resolution change complete")

    async def stop(self):
        """Stop the pipeline. Public method that acquires the lock."""
        try:
            # Add a timeout to prevent hanging indefinitely
            await asyncio.wait_for(self._stop_with_lock(), timeout=10.0)
        except asyncio.TimeoutError:
            logging.error("Timeout occurred while stopping StreamDiffusion pipeline")
            # Force cleanup even if timeout occurred
            self.pipe = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logging.warning(f"Error during CUDA cache cleanup: {e}")
        except Exception as e:
            logging.error(f"Error stopping StreamDiffusion pipeline: {e}")
            self.pipe = None

    async def _stop_with_lock(self):
        """Helper method to acquire lock and stop."""
        async with self._pipeline_lock:
            await self._stop_locked()

    async def _stop_locked(self):
        """Internal method to stop the pipeline. Must be called while holding the pipeline lock."""
        logging.info("Stopping StreamDiffusion pipeline")
        
        # Clear the frame queue with timeout protection
        try:
            queue_clear_count = 0
            while not self.frame_queue.empty() and queue_clear_count < 100:  # Limit iterations
                try:
                    self.frame_queue.get_nowait()
                    queue_clear_count += 1
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logging.warning(f"Error clearing frame from queue: {e}")
                    break
        except Exception as e:
            logging.warning(f"Error during frame queue cleanup: {e}")
        
        # Clear pipeline tensors and delete the pipeline
        if self.pipe is not None:
            try:
                # Set self.pipe to None
                self.pipe = None
                logging.info("Pipeline deleted successfully")
            except Exception as e:
                logging.warning(f"Error deleting pipeline: {e}")
                self.pipe = None
        
        # Force CUDA cache clearing with timeout protection
        if torch.cuda.is_available():
            try:
                # Run CUDA operations in a separate thread with timeout
                await asyncio.wait_for(
                    asyncio.to_thread(self._cuda_cleanup), 
                    timeout=5.0  # 5 second timeout for CUDA cleanup
                )
                logging.info("CUDA cleanup completed successfully")
            except asyncio.TimeoutError:
                logging.warning("CUDA cleanup timed out, continuing anyway")
            except Exception as e:
                logging.warning(f"Error during CUDA cleanup: {e}")

    def _cuda_cleanup(self):
        """Synchronous CUDA cleanup operations."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_streamdiffusion_sync(params: StreamDiffusionParams):
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
    )
    pipe.prepare(
        prompt=params.prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
    )
    return pipe

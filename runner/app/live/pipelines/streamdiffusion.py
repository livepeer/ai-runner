import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, cast

import torch
from streamdiffusion import StreamDiffusionWrapper

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput

from .streamdiffusion_params import StreamDiffusionParams, ControlNetConfig

class StreamDiffusion(Pipeline):
    def __init__(self):
        super().__init__()
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.params: Optional[StreamDiffusionParams] = None
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
        img_tensor = cast(torch.Tensor, self.pipe.stream.image_processor.denormalize(img_tensor))
        img_tensor = self.pipe.preprocess_image(img_tensor)

        if self.params and self.params.controlnets:
            for i, cn in enumerate(self.params.controlnets):
                if cn.enabled and cn.conditioning_scale > 0:
                    self.pipe.update_control_image(i, img_tensor)

        if self.first_frame:
            self.first_frame = False
            for _ in range(self.pipe.batch_size):
                self.pipe(image=img_tensor)

        out_tensor = self.pipe(image=img_tensor)
        if isinstance(out_tensor, list):
            out_tensor = out_tensor[0]

        # The output tensor from the wrapper is (1, C, H, W), and the encoder expects (1, H, W, C).
        return out_tensor.permute(0, 2, 3, 1)

    async def get_processed_video_frame(self) -> VideoOutput:
        return await self.frame_queue.get()

    async def update_params(self, **params):
        new_params = StreamDiffusionParams(**params)
        if new_params == self.params:
            logging.info("No parameters changed")
            return

        async with self._pipeline_lock:
            try:
                if await self._update_params_dynamic(new_params):
                    return
            except Exception as e:
                logging.error(f"Error updating parameters dynamically: {e}")

            logging.info(f"Resetting pipeline for params change")

            self.pipe = None
            try:
                self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
            except Exception:
                logging.error(f"Error resetting pipeline, reloading with previous params", exc_info=True)
                new_params = self.params or StreamDiffusionParams()
                self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)

            self.params = new_params
            self.applied_controlnets = new_params.controlnets
            self.first_frame = True

    async def _update_params_dynamic(self, new_params: StreamDiffusionParams):
        if self.pipe is None:
            return False

        updatable_params = {
            'num_inference_steps', 'guidance_scale', 'delta', 't_index_list',
            'prompt', 'prompt_interpolation_method', 'normalize_prompt_weights', 'negative_prompt',
            'seed', 'seed_interpolation_method', 'normalize_seed_weights',
            'controlnets', # handled separately below
        }

        update_kwargs = {}
        curr_params = self.params.model_dump() if self.params else {}
        for key, new_value in new_params.model_dump().items():
            curr_value = curr_params.get(key, None)
            if new_value == curr_value:
                continue
            elif key not in updatable_params:
                logging.info(f"Non-updatable parameter changed: {key}")
                return False

            # at this point, we know it's an updatable parameter that changed
            if key == 'prompt':
                update_kwargs['prompt_list'] = [(new_value, 1.0)] if isinstance(new_value, str) else new_value
            elif key == 'seed':
                update_kwargs['seed_list'] = [(new_value, 1.0)] if isinstance(new_value, int) else new_value
            elif key == 'controlnets':
                update_kwargs['controlnet_config'] = _prepare_controlnet_configs(new_params)
            else:
                update_kwargs[key] = new_value

        logging.info(f"Updating parameters dynamically update_kwargs={update_kwargs}")

        if update_kwargs:
            self.pipe.update_stream_params(**update_kwargs)

        self.params = new_params
        self.first_frame = True
        return True

    async def stop(self):
        async with self._pipeline_lock:
            self.pipe = None
            self.params = None
            self.frame_queue = asyncio.Queue()


def _prepare_controlnet_configs(params: StreamDiffusionParams) -> Optional[List[Dict[str, Any]]]:
    """Prepare ControlNet configurations for wrapper"""
    if not params.controlnets:
        return None

    controlnet_configs = []
    for cn_config in params.controlnets:
        if not cn_config.enabled:
            continue

        preprocessor_params = (cn_config.preprocessor_params or {}).copy()

        # Inject preprocessor-specific parameters
        if cn_config.preprocessor == "depth_tensorrt":
            preprocessor_params.update({
                "engine_path": "./engines/depth-anything/depth_anything_v2_vits.engine",
            })
        elif cn_config.preprocessor == "pose_tensorrt":
            confidence_threshold = preprocessor_params.pop("confidence_threshold", 0.5)

            engine_path = f"./engines/pose/yolo_nas_pose_l_{confidence_threshold}.engine"
            if not os.path.exists(engine_path):
                raise ValueError(f"Invalid confidence threshold: {confidence_threshold}")

            preprocessor_params.update({
                "engine_path": engine_path,
            })

        controlnet_config = {
            'model_id': cn_config.model_id,
            'preprocessor': cn_config.preprocessor,
            'conditioning_scale': cn_config.conditioning_scale,
            'enabled': cn_config.enabled,
            'preprocessor_params': preprocessor_params,
            'control_guidance_start': cn_config.control_guidance_start,
            'control_guidance_end': cn_config.control_guidance_end,
        }
        controlnet_configs.append(controlnet_config)

    return controlnet_configs


def load_streamdiffusion_sync(params: StreamDiffusionParams, min_batch_size = 1, max_batch_size = 4, engine_dir = "engines", build_engines_if_missing = False):
    # Prepare ControlNet configuration
    controlnet_config = _prepare_controlnet_configs(params)

    pipe = StreamDiffusionWrapper(
        model_id_or_path=params.model_id,
        t_index_list=params.t_index_list,
        # min_batch_size=min_batch_size,
        # max_batch_size=max_batch_size,
        lora_dict=params.lora_dict,
        mode="img2img",
        output_type="pt",
        lcm_lora_id=params.lcm_lora_id,
        frame_buffer_size=1,
        width=params.width,
        height=params.height,
        warmup=10,
        acceleration=params.acceleration,
        do_add_noise=params.do_add_noise,
        use_lcm_lora=params.use_lcm_lora,
        enable_similar_image_filter=params.enable_similar_image_filter,
        similar_image_filter_threshold=params.similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=params.similar_image_filter_max_skip_frame,
        use_denoising_batch=params.use_denoising_batch,
        seed=params.seed if isinstance(params.seed, int) else params.seed[0][0],
        normalize_seed_weights=params.normalize_seed_weights,
        normalize_prompt_weights=params.normalize_prompt_weights,
        use_controlnet=bool(controlnet_config),
        controlnet_config=controlnet_config,
        engine_dir=engine_dir,
        build_engines_if_missing=build_engines_if_missing,
    )

    pipe.prepare(
        prompt=params.prompt,
        prompt_interpolation_method=params.prompt_interpolation_method,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        delta=params.delta,
        seed_list=[(params.seed, 1.0)] if isinstance(params.seed, int) else params.seed,
        seed_interpolation_method=params.seed_interpolation_method,
    )
    return pipe

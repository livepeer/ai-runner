import os
import logging
import asyncio
from typing import Dict, List, Literal, Optional, Any, Tuple, cast

import torch
from pydantic import BaseModel
from streamdiffusion import StreamDiffusionWrapper

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT

class ControlNetConfig(BaseModel):
    """ControlNet configuration model"""
    model_id: str
    conditioning_scale: float = 1.0
    preprocessor: Optional[str] = None
    preprocessor_params: Optional[Dict[str, Any]] = None
    enabled: bool = True
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0


class StreamDiffusionParams(BaseModel):
    class Config:
        extra = "forbid"

    # Model configuration
    model_id: str = "stabilityai/sd-turbo"

    # Generation parameters
    prompt: str | List[Tuple[str, float]] = "an anime render of a girl with purple hair, masterpiece"
    prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    negative_prompt: str = "blurry, low quality, flat, 2d"
    guidance_scale: float = 1.0
    delta: float = 0.7
    num_inference_steps: int = 50
    t_index_list: List[int] = [12, 20, 32]

    # Image dimensions
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    # LoRA settings
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"

    # Acceleration settings
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"

    # Processing settings
    use_denoising_batch: bool = True
    do_add_noise: bool = True
    seed: int | List[Tuple[int, float]] = 789
    seed_interpolation_method: Literal["linear", "slerp"] = "linear"

    # Similar image filter settings
    enable_similar_image_filter: bool = False
    similar_image_filter_threshold: float = 0.98
    similar_image_filter_max_skip_frame: int = 10

    # ControlNet settings
    controlnets: Optional[List[ControlNetConfig]] = [
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-openpose-diffusers",
            conditioning_scale=0.711,
            preprocessor="pose_tensorrt",
            preprocessor_params={
                "confidence_threshold": 0.5,
            },
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        ),
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-hed-diffusers",
            conditioning_scale=0.2,
            preprocessor="soft_edge",
            preprocessor_params={},
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        ),
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-canny-diffusers",
            conditioning_scale=0.2,
            preprocessor="canny",
            preprocessor_params={
                "low_threshold": 100,
                "high_threshold": 200
            },
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        ),
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-depth-diffusers",
            conditioning_scale=0.5,
            preprocessor="depth_tensorrt",
            preprocessor_params={},
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        ),
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-color-diffusers",
            conditioning_scale=0.2,
            preprocessor="passthrough",
            preprocessor_params={},
            enabled=True,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
        )
    ]


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

        # Noop if ControlNets are not enabled
        self.pipe.update_control_image_efficient(img_tensor)

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
        async with self._pipeline_lock:
            new_params = StreamDiffusionParams(**params)
            if new_params == self.params:
                logging.info("No parameters changed")
                return

            if self.pipe is not None:
                updatable_params = {
                    'num_inference_steps', 'guidance_scale', 'delta', 't_index_list', 'seed', 'prompt', 'prompt_interpolation_method', 'negative_prompt', 'seed_interpolation_method'
                }

                only_updatable_changed = True
                curr_params = self.params.model_dump() if self.params else {}
                for key, new_value in new_params.model_dump().items():
                    curr_value = curr_params.get(key, None)
                    if key not in updatable_params and new_value != curr_value:
                        only_updatable_changed = False
                        logging.info(f"Non-updatable parameter changed: {key}")
                        break
                    elif key == 't_index_list' and len(new_value) != len(curr_value or []):
                        only_updatable_changed = False
                        logging.info(f"Non-updatable parameter changed: length of t_index_list")
                        break

                if only_updatable_changed:
                    logging.info("Updating parameters via update_stream_params")

                    update_kwargs = {
                        k: v for k, v
                        in new_params.model_dump().items()
                        if k in updatable_params and v != getattr(self.params, k)
                    }

                    # Some fields are named/typed differently from our params in the update_stream_params method
                    if 'prompt' in update_kwargs:
                        prompt = update_kwargs.pop('prompt')
                        update_kwargs['prompt_list'] = [(prompt, 1.0)] if isinstance(prompt, str) else prompt
                    if 'prompt_interpolation_method' in update_kwargs:
                        update_kwargs['interpolation_method'] = update_kwargs.pop('prompt_interpolation_method')
                    if 'seed' in update_kwargs:
                        seed = update_kwargs.pop('seed')
                        update_kwargs['seed_list'] = [(seed, 1.0)] if isinstance(seed, int) else seed

                    try:
                        self.pipe.update_stream_params(**update_kwargs)
                        self.params = new_params
                        return
                    except Exception as e:
                        logging.error(f"Error updating parameters dynamically: {e}")

            logging.info(f"Resetting pipeline for params change")

            self.pipe = None
            self.pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
            self.params = new_params
            self.first_frame = True

    async def stop(self):
        async with self._pipeline_lock:
            self.pipe = None
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
        if cn_config.preprocessor in ["canny", "hed", "soft_edge", "passthrough"]:
            # no enforced params
            pass
        elif cn_config.preprocessor == "depth_tensorrt":
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
        else:
            raise ValueError(f"Unrecognized preprocessor: {cn_config.preprocessor}")

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


def load_streamdiffusion_sync(params: StreamDiffusionParams, engine_dir = "engines", build_engines_if_missing = False):
    # Prepare ControlNet configuration
    controlnet_config = _prepare_controlnet_configs(params)

    pipe = StreamDiffusionWrapper(
        model_id_or_path=params.model_id,
        t_index_list=params.t_index_list,
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
        use_controlnet=bool(controlnet_config),
        controlnet_config=controlnet_config,
        engine_dir=engine_dir,
        build_engines_if_missing=build_engines_if_missing,
    )

    pipe.prepare(
        prompt=params.prompt,
        interpolation_method=params.prompt_interpolation_method,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        delta=params.delta,
        seed_list=[(params.seed, 1.0)] if isinstance(params.seed, int) else params.seed,
        seed_interpolation_method=params.seed_interpolation_method,
    )
    return pipe

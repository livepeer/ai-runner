import logging
import asyncio
from typing import Dict, List, Literal, Optional, Any

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
    prompt: str = "an anime render of a girl with purple hair, masterpiece"
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
    seed: int = 789

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
        seed=params.seed,
        use_controlnet=bool(controlnet_config),
        controlnet_config=controlnet_config,
        engine_dir=engine_dir,
        build_engines_if_missing=build_engines_if_missing,
    )

    pipe.prepare(
        prompt=params.prompt,
        negative_prompt=params.negative_prompt,
        num_inference_steps=params.num_inference_steps,
        guidance_scale=params.guidance_scale,
        delta=params.delta,
    )
    return pipe

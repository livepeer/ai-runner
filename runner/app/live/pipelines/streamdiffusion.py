import os
import logging
import asyncio
from typing import Dict, List, Literal, Optional, Any, Tuple, cast

import torch
from pydantic import BaseModel, Field, model_validator
from streamdiffusion import StreamDiffusionWrapper
from streamdiffusion.controlnet.preprocessors import list_preprocessors

from .interface import Pipeline
from trickle import VideoFrame, VideoOutput
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT

AVAILABLE_PREPROCESSORS = list_preprocessors()

class ControlNetConfig(BaseModel):
    """ControlNet configuration model"""
    model_id: Literal[
        "thibaud/controlnet-sd21-openpose-diffusers",
        "thibaud/controlnet-sd21-hed-diffusers",
        "thibaud/controlnet-sd21-canny-diffusers",
        "thibaud/controlnet-sd21-depth-diffusers",
        "thibaud/controlnet-sd21-color-diffusers"
    ]
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
    model_id: Literal[
        "stabilityai/sd-turbo",
        "KBlueLeaf/kohaku-v2.1",
    ] = "stabilityai/sd-turbo"

    # Generation parameters
    prompt: str | List[Tuple[str, float]] = "an anime render of a girl with purple hair, masterpiece"
    prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    normalize_prompt_weights: bool = True
    negative_prompt: str = "blurry, low quality, flat, 2d"
    guidance_scale: float = 1.0
    delta: float = 0.7
    num_inference_steps: int = 50
    t_index_list: List[int] = [12, 20, 32]

    # Image dimensions
    width: int = Field(default=DEFAULT_WIDTH, ge=384, le=1024, multiple_of=64)
    height: int = Field(default=DEFAULT_HEIGHT, ge=384, le=1024, multiple_of=64)

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
    normalize_seed_weights: bool = True

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
            preprocessor_params={},
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

    @model_validator(mode="after")
    @staticmethod
    def check_t_index_list(model: "StreamDiffusionParams") -> "StreamDiffusionParams":
        if not (1 <= len(model.t_index_list) <= 4):
            raise ValueError("t_index_list must have between 1 and 4 elements")

        for i, value in enumerate(model.t_index_list):
            if not (0 <= value <= model.num_inference_steps):
                raise ValueError(
                    f"Each t_index_list value must be between 0 and num_inference_steps ({model.num_inference_steps}). Found {value} at index {i}."
                )

        for i in range(1, len(model.t_index_list)):
            curr, prev = model.t_index_list[i], model.t_index_list[i - 1]
            if curr < prev:
                raise ValueError(f"t_index_list must be in non-decreasing order. {curr} < {prev}")

        # Check for duplicate controlnet model_ids
        if model.controlnets:
            seen_model_ids = set()
            for cn in model.controlnets:
                if cn.model_id in seen_model_ids:
                    raise ValueError(f"Duplicate controlnet model_id: {cn.model_id}")
                seen_model_ids.add(cn.model_id)

        return model


class StreamDiffusion(Pipeline):
    def __init__(self):
        super().__init__()
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.params: Optional[StreamDiffusionParams] = None
        self.applied_controlnets: Optional[List[ControlNetConfig]] = None
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
        patched_controlnets = None
        curr_params = self.params.model_dump() if self.params else {}
        for key, new_value in new_params.model_dump().items():
            curr_value = curr_params.get(key, None)
            if new_value == curr_value:
                continue
            elif key not in updatable_params:
                logging.info(f"Non-updatable parameter changed: {key}")
                return False
            elif key == 'controlnets':
                patched_controlnets = _compute_controlnet_patch(
                    self.applied_controlnets, new_params.controlnets
                )
                if patched_controlnets is None:
                    logging.info("Non-updatable parameter changed: controlnets")
                    return False
                # do not add controlnets to update_kwargs
                continue

            # at this point, we know it's an updatable parameter that changed
            if key == 'prompt':
                update_kwargs['prompt_list'] = [(new_value, 1.0)] if isinstance(new_value, str) else new_value
            elif key == 'seed':
                update_kwargs['seed_list'] = [(new_value, 1.0)] if isinstance(new_value, int) else new_value
            else:
                update_kwargs[key] = new_value

        logging.info(
            f"Updating parameters dynamically update_kwargs={update_kwargs} patched_controlnets={patched_controlnets}"
        )

        if update_kwargs:
            self.pipe.update_stream_params(**update_kwargs)
        if patched_controlnets:
            applied_controlnets = self.applied_controlnets or []
            for i, patched in enumerate(patched_controlnets):
                old_scale = applied_controlnets[i].conditioning_scale
                if patched.conditioning_scale != old_scale:
                    self.pipe.update_controlnet_scale(i, patched.conditioning_scale)
            # Only update the applied_controlnets if we actually patched something
            self.applied_controlnets = patched_controlnets

        self.params = new_params
        self.first_frame = True
        return True

    async def stop(self):
        async with self._pipeline_lock:
            self.pipe = None
            self.params = None
            self.applied_controlnets = None
            self.frame_queue = asyncio.Queue()


def _compute_controlnet_patch(
    curr: Optional[List[ControlNetConfig]],
    new: Optional[List[ControlNetConfig]],
) -> Optional[List[ControlNetConfig]]:
    """
    Reconcile a controlnet update as a patch to the currently applied list. Returns None if the new controlnets cannot be
    patched without a full reload. This is only possible if there are no new controlnets or config changes compared to
    the currently applied list. Returns a list of patched controlnets if the update can be applied dynamically.
    """
    curr = curr or []
    new = new or []

    index_by_model: Dict[str, int] = {cn.model_id: i for i, cn in enumerate(curr)}

    # Start with 0 scales for every current controlnet and apply scales of the new params below
    patched_list = [
        cn.model_copy(deep=True, update={"conditioning_scale": 0.0}) for cn in curr
    ]
    for new_cn in new:
        if not new_cn.enabled or new_cn.conditioning_scale == 0:
            # We can ignore disabled controlnets here and keep them out of the patched list (or with scale 0)
            continue

        idx = index_by_model.get(new_cn.model_id)
        if idx is None:
            logging.info(
                f"Controlnet config changed, adding new controlnet. model_id={new_cn.model_id}"
            )
            return None

        curr_cn = curr[idx]
        if not curr_cn.enabled:
            logging.info(
                f"Controlnet config changed, enabling controlnet. model_id={new_cn.model_id}"
            )
            return None

        patched_cn = curr_cn.model_copy(
            update={"conditioning_scale": new_cn.conditioning_scale}
        )
        if patched_cn != new_cn:
            logging.info(
                f"Controlnet config changed, params updated. model_id={new_cn.model_id} previous={curr_cn} new={new_cn}"
            )
            return None

        patched_list[idx] = patched_cn

    return patched_list

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
        elif cn_config.preprocessor not in AVAILABLE_PREPROCESSORS:
            raise ValueError(f"Unrecognized preprocessor: '{cn_config.preprocessor}'. Must be one of {AVAILABLE_PREPROCESSORS}")

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
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
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

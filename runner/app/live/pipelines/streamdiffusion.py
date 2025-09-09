import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, cast

import torch
from streamdiffusion import StreamDiffusionWrapper
from PIL import Image
from io import BytesIO
import aiohttp

from .interface import Pipeline
from .loading_overlay import LoadingOverlayRenderer
from trickle import VideoFrame, VideoOutput

from .streamdiffusion_params import StreamDiffusionParams, IPAdapterConfig, get_model_type, IPADAPTER_SUPPORTED_TYPES

class StreamDiffusion(Pipeline):
    def __init__(self):
        super().__init__()
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.params: Optional[StreamDiffusionParams] = None
        self.first_frame = True
        self.frame_queue: asyncio.Queue[VideoOutput] = asyncio.Queue()
        self._pipeline_lock = asyncio.Lock()  # Protects pipeline initialization/reinitialization
        self._overlay_renderer = LoadingOverlayRenderer()
        self._cached_style_image_tensor: Optional[torch.Tensor] = None
        self._cached_style_image_url: Optional[str] = None

    async def initialize(self, **params):
        logging.info(f"Initializing StreamDiffusion pipeline with params: {params}")
        await self.update_params(**params)
        logging.info("Pipeline initialization complete")

    async def put_video_frame(self, frame: VideoFrame, request_id: str):
        if self.params is None:
            raise RuntimeError("Pipeline not initialized")

        async with self._pipeline_lock:
            loading_frame = await self._overlay_renderer.render_if_active(self.params.width, self.params.height)
            if loading_frame is not None:
                output = VideoOutput(frame, request_id, is_loading_frame=True).replace_tensor(loading_frame)
            else:
                out_tensor = await asyncio.to_thread(self.process_tensor_sync, frame.tensor)
                output = VideoOutput(frame, request_id).replace_tensor(out_tensor)
                self._overlay_renderer.update_last_frame(out_tensor)

        await self.frame_queue.put(output)

    def process_tensor_sync(self, img_tensor: torch.Tensor):
        if self.pipe is None:
            raise RuntimeError("Pipeline not initialized")

        # The incoming frame.tensor is (B, H, W, C) in range [-1, 1] while the
        # VaeImageProcessor inside the wrapper expects (B, C, H, W) in [0, 1].
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = cast(
            torch.Tensor, self.pipe.stream.image_processor.denormalize(img_tensor)
        )
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

        if out_tensor.dim() == 3:
            # Workaround as the NSFW fallback image is coming without the batch dimension
            out_tensor = out_tensor.unsqueeze(0)
            return out_tensor

        # The output tensor from the wrapper is (1, C, H, W), and the encoder expects (1, H, W, C).
        out_bhwc = out_tensor.permute(0, 2, 3, 1)
        return out_bhwc

    async def get_processed_video_frame(self) -> VideoOutput:
        return await self.frame_queue.get()

    async def update_params(self, **params):
        new_params = StreamDiffusionParams(**params)
        if new_params == self.params:
            logging.info("No parameters changed")
            return

        self._overlay_renderer.set_show_overlay(new_params.show_reloading_frame)

        # Pre-fetch the style image before locking. This raises any errors early (e.g. invalid URL or image) and also
        # allows us to fetch the style image without blocking inference with the lock.
        if (
            new_params.ip_adapter_style_image_url
            and new_params.ip_adapter_style_image_url != self._cached_style_image_url
        ):
            await self._fetch_style_image(new_params.ip_adapter_style_image_url)

        async with self._pipeline_lock:
            try:
                if await self._update_params_dynamic(new_params):
                    return
            except Exception as e:
                logging.error(f"Error updating parameters dynamically: {e}")

        logging.info(f"Resetting pipeline for params change")

        try:
            await self._overlay_renderer.prewarm(new_params.width, new_params.height)
        except Exception:
            logging.debug("Failed to prewarm loading overlay caches", exc_info=True)

        async with self._pipeline_lock:
            # Clear the pipeline while loading the new one. The loading overlay will be shown while this is happening.
            self.pipe = None
            prev_params = self.params
            self._overlay_renderer.begin_reload()

        new_pipe: Optional[StreamDiffusionWrapper] = None
        try:
            new_pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
        except Exception:
            logging.error(f"Error resetting pipeline, reloading with previous params", exc_info=True)
            try:
                new_params = prev_params or StreamDiffusionParams()
                new_pipe = await asyncio.to_thread(load_streamdiffusion_sync, new_params)
            except Exception:
                logging.exception("Failed to reload pipeline with fallback params", stack_info=True)
                raise

        async with self._pipeline_lock:
            self.pipe = new_pipe
            self.params = new_params
            self.first_frame = True
            self._overlay_renderer.end_reload()

            if new_params.ip_adapter and new_params.ip_adapter.enabled:
                await self._update_style_image(new_params)
                # no-op update prompt to cause an IPAdapter reload
                self.pipe.update_stream_params(prompt_list=self.pipe.stream._param_updater.get_current_prompts())

    async def _update_params_dynamic(self, new_params: StreamDiffusionParams):
        if self.pipe is None:
            return False

        updatable_params = {
            'num_inference_steps', 'guidance_scale', 'delta', 't_index_list',
            'prompt', 'prompt_interpolation_method', 'normalize_prompt_weights', 'negative_prompt',
            'seed', 'seed_interpolation_method', 'normalize_seed_weights',
            'use_safety_checker', 'safety_checker_threshold', 'controlnets',
            'ip_adapter', 'ip_adapter_style_image_url', 'show_reloading_frame'
        }

        update_kwargs = {}
        curr_params = self.params.model_dump() if self.params else {}
        changed_ipadapter = False
        for key, new_value in new_params.model_dump().items():
            curr_value = curr_params.get(key, None)
            if new_value == curr_value:
                continue
            elif key not in updatable_params:
                logging.info(f"Non-updatable parameter changed: {key}")
                return False
            elif key == 'show_reloading_frame':
                # Handled by us in update_params, not a config from the lib
                continue

            # at this point, we know it's an updatable parameter that changed
            if key == 'prompt':
                update_kwargs['prompt_list'] = [(new_value, 1.0)] if isinstance(new_value, str) else new_value
            elif key == 'seed':
                update_kwargs['seed_list'] = [(new_value, 1.0)] if isinstance(new_value, int) else new_value
            elif key == 'controlnets':
                update_kwargs['controlnet_config'] = _prepare_controlnet_configs(new_params)
            elif key == 'ip_adapter':
                # Check if only dynamic params have changed
                only_dynamic_changes = curr_params.get('ip_adapter') or IPAdapterConfig().model_dump()
                for k in ['enabled', 'scale', 'weight_type']:
                    only_dynamic_changes[k] = new_value[k]
                if new_value != only_dynamic_changes:
                    return False

                update_kwargs['ipadapter_config'] = _prepare_ipadapter_configs(new_params)
                changed_ipadapter = True
            elif key == 'ip_adapter_style_image_url':
                # Do not set on update_kwargs, we'll update it separately.
                changed_ipadapter = True
            else:
                update_kwargs[key] = new_value

        logging.info(f"Updating parameters dynamically update_kwargs={update_kwargs}")

        if update_kwargs:
            self.pipe.update_stream_params(**update_kwargs)
        if changed_ipadapter:
            await self._update_style_image(new_params)
            # no-op update prompt to cause an IPAdapter reload
            self.pipe.update_stream_params(prompt_list=self.pipe.stream._param_updater.get_current_prompts())

        self.params = new_params
        self.first_frame = True
        return True

    async def _update_style_image(self, params: StreamDiffusionParams) -> None:
        assert self.pipe is not None

        style_image_url = params.ip_adapter_style_image_url
        ipadapter_enabled = params.ip_adapter is not None and params.ip_adapter.enabled
        if not ipadapter_enabled:
            return

        if style_image_url and style_image_url != self._cached_style_image_url:
            await self._fetch_style_image(style_image_url)

        if self._cached_style_image_tensor is not None:
            self.pipe.update_style_image(self._cached_style_image_tensor)
        else:
            logging.warning("[IPAdapter] No cached style image tensor; skipping style image update")

    async def _fetch_style_image(self, style_image_url: str):
        """
        Pre-fetches the style image and caches it in self._cached_style_image_tensor.

        If the pipe is not initialized, this just validates that the image in the URL is valid and return.
        """
        image = await _load_image_from_url(style_image_url)
        if self.pipe is None:
            return

        tensor = self.pipe.preprocess_image(image)
        self._cached_style_image_tensor = tensor
        self._cached_style_image_url = style_image_url

    async def stop(self):
        async with self._pipeline_lock:
            self.pipe = None
            self.params = None
            self.frame_queue = asyncio.Queue()
            self._overlay_renderer.end_reload()
            self._overlay_renderer.reset_session(0.0)


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

def _prepare_ipadapter_configs(params: StreamDiffusionParams) -> Optional[Dict[str, Any]]:
    """Prepare IPAdapter configurations for wrapper"""
    if not params.ip_adapter:
        return None

    ip_cfg = params.ip_adapter.model_copy()
    if ip_cfg.ipadapter_model_path:
        logging.warning(f"[IPAdapter] ipadapter_model_path is deprecated and will be ignored. Use type instead.")
    if ip_cfg.image_encoder_path:
        logging.warning(f"[IPAdapter] image_encoder_path is deprecated and will be ignored. Use type instead.")

    model_type = get_model_type(params.model_id)
    dir = 'sdxl_models' if model_type == 'sdxl' else 'models'

    if not ip_cfg.ipadapter_model_path:
        match ip_cfg.type:
            case 'regular':
                ip_cfg.ipadapter_model_path = f"h94/IP-Adapter/{dir}/ip-adapter_{model_type}.bin" # type: ignore
            case 'faceid':
                ip_cfg.ipadapter_model_path = f"h94/IP-Adapter-FaceID/ip-adapter-faceid_{model_type}.bin" # type: ignore
    if not ip_cfg.image_encoder_path:
        ip_cfg.image_encoder_path = f"h94/IP-Adapter/{dir}/image_encoder" # type: ignore

    if not ip_cfg.enabled:
        # Enabled flag is ignored, so we set scale to 0.0 to disable it.
        ip_cfg.scale = 0.0

    return ip_cfg.model_dump()


def load_streamdiffusion_sync(
    params: StreamDiffusionParams,
    min_batch_size=1,
    max_batch_size=4,
    engine_dir="engines",
    build_engines=False,
) -> StreamDiffusionWrapper:
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
        use_controlnet=True,
        controlnet_config=_prepare_controlnet_configs(params),
        use_ipadapter=get_model_type(params.model_id) in IPADAPTER_SUPPORTED_TYPES,
        ipadapter_config=_prepare_ipadapter_configs(params),
        engine_dir=engine_dir,
        build_engines_if_missing=build_engines,
        compile_engines_only=build_engines,
        use_safety_checker=params.use_safety_checker,
        safety_checker_threshold=params.safety_checker_threshold,
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


async def _load_image_from_url(url: str) -> Image.Image:
    if not (url.startswith('http://') or url.startswith('https://')):
        raise ValueError(f"Invalid image URL: {url}")

    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
    return Image.open(BytesIO(data)).convert('RGB')

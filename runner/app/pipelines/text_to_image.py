import logging
import os
from enum import Enum
from typing import List, Optional, Tuple

import PIL
import torch

from diffusers import (
    AutoPipelineForText2Image,
    EulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.models import AutoencoderKL
from huggingface_hub import file_download, hf_hub_download
from safetensors.torch import load_file

from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    LoraLoader,
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model,
    split_prompt,
)
from app.utils.errors import InferenceError

from app.pipelines.device_maps import (
    LPFluxPipeline,
)

logger = logging.getLogger(__name__)


class ModelName(Enum):
    """Enumeration mapping model names to their corresponding IDs."""

    SDXL_LIGHTNING = "ByteDance/SDXL-Lightning"
    SD3 = "stabilityai/stable-diffusion-3"
    REALISTIC_VISION_V6 = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    FLUX_1_SCHNELL = "black-forest-labs/FLUX.1-schnell"
    FLUX_1_DEV = "black-forest-labs/FLUX.1-dev"

    @classmethod
    def list(cls):
        """Return a list of all model IDs."""
        return [model.value for model in cls]


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load the fp16 variant if fp16 'safetensors' files are present in the cache.
        # NOTE: Exception for SDXL-Lightning model: despite having fp16 'safetensors'
        # files, they are not named according to the standard convention.
        has_fp16_variant = (
            any(
                ".fp16.safetensors" in fname
                for _, _, files in os.walk(folder_path)
                for fname in files
            )
            or ModelName.SDXL_LIGHTNING.value in model_id
        )
        if torch_device.type != "cpu" and has_fp16_variant:
            logger.info("TextToImagePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("TextToImagePipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        # Load VAE for specific models.
        if ModelName.REALISTIC_VISION_V6.value in model_id:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
            kwargs["vae"] = vae

        # Special case SDXL-Lightning because the unet for SDXL needs to be swapped
        if ModelName.SDXL_LIGHTNING.value in model_id:
            base = "stabilityai/stable-diffusion-xl-base-1.0"

            # ByteDance/SDXL-Lightning-2step
            if "2step" in model_id:
                unet_id = "sdxl_lightning_2step_unet"
            # ByteDance/SDXL-Lightning-4step
            elif "4step" in model_id:
                unet_id = "sdxl_lightning_4step_unet"
            # ByteDance/SDXL-Lightning-8step
            elif "8step" in model_id:
                unet_id = "sdxl_lightning_8step_unet"
            else:
                # Default to 8step
                unet_id = "sdxl_lightning_8step_unet"

            unet_config = UNet2DConditionModel.load_config(
                pretrained_model_name_or_path=base,
                subfolder="unet",
                cache_dir=kwargs["cache_dir"],
            )
            unet = UNet2DConditionModel.from_config(unet_config).to(
                torch_device, kwargs["torch_dtype"]
            )
            unet.load_state_dict(
                load_file(
                    hf_hub_download(
                        ModelName.SDXL_LIGHTNING.value,
                        f"{unet_id}.safetensors",
                        cache_dir=kwargs["cache_dir"],
                    ),
                    device=str(torch_device),
                )
            )

            self.ldm = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, **kwargs
            ).to(torch_device)

            self.ldm.scheduler = EulerDiscreteScheduler.from_config(
                self.ldm.scheduler.config, timestep_spacing="trailing"
            )
        elif ModelName.SD3.value in model_id:
            self.ldm = StableDiffusion3Pipeline.from_pretrained(model_id, **kwargs).to(
                torch_device
            )
            kwargs["torch_dtype"] = torch.bfloat16
        elif (
            ModelName.FLUX_1_SCHNELL.value in model_id
            or ModelName.FLUX_1_DEV.value in model_id
        ):
            # Decrease precision to preven OOM errors.
            kwargs["torch_dtype"] = torch.bfloat16
            self.ldm = LPFluxPipeline(model_id, os.environ.get("DEVICE_MAP", ""), torch_device, **kwargs)
        else:
            self.ldm = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to(
                torch_device
            )

        if os.environ.get("TORCH_COMPILE"):
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            self.ldm.unet.to(memory_format=torch.channels_last)
            self.ldm.vae.to(memory_format=torch.channels_last)

            self.ldm.unet = torch.compile(
                self.ldm.unet, mode="max-autotune", fullgraph=True
            )
            self.ldm.vae.decode = torch.compile(
                self.ldm.vae.decode, mode="max-autotune", fullgraph=True
            )

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        deepcache_enabled = os.getenv("DEEPCACHE", "").strip().lower() == "true"
        if sfast_enabled and deepcache_enabled:
            logger.warning(
                "Both 'SFAST' and 'DEEPCACHE' are enabled. This is not recommended "
                "as it may lead to suboptimal performance. Please disable one of them."
            )

        if sfast_enabled:
            logger.info(
                "TextToImagePipeline will be dynamically compiled with stable-fast for "
                "%s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # TODO: Not yet supported for TextToImagePipeline.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                logger.warning(
                    "The 'SFAST_WARMUP' flag is not yet supported for the "
                    "TextToImagePipeline and will be ignored. As a result the first "
                    "call may be slow if 'SFAST' is enabled."
                )

        if deepcache_enabled and not (
            is_lightning_model(model_id) or is_turbo_model(model_id)
        ):
            logger.info(
                "TextToImagePipeline will be optimized with DeepCache for %s",
                model_id,
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            self.ldm = enable_deepcache(self.ldm)
        elif deepcache_enabled:
            logger.warning(
                "DeepCache is not supported for Lightning or Turbo models. "
                "TextToImagePipeline will NOT be optimized with DeepCache for %s",
                model_id,
            )

        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

        self._lora_loader = LoraLoader(self.ldm)

    def __call__(
        self, prompt: str, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)
        loras_json = kwargs.pop("loras", "")

        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        # Dynamically (un)load LoRas.
        if not loras_json:
            self._lora_loader.disable_loras()
        else:
            self._lora_loader.load_loras(loras_json)

        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        if (
            self.model_id == "stabilityai/sdxl-turbo"
            or self.model_id == "stabilityai/sd-turbo"
        ):
            # SD turbo models were trained without guidance_scale so
            # it should be set to 0
            kwargs["guidance_scale"] = 0.0
        elif ModelName.SDXL_LIGHTNING.value in self.model_id:
            # SDXL-Lightning models should have guidance_scale = 0 and use
            # the correct number of inference steps for the unet checkpoint loaded
            kwargs["guidance_scale"] = 0.0

            if "2step" in self.model_id:
                kwargs["num_inference_steps"] = 2
            elif "4step" in self.model_id:
                kwargs["num_inference_steps"] = 4
            elif "8step" in self.model_id:
                kwargs["num_inference_steps"] = 8
            else:
                # Default to 8step
                kwargs["num_inference_steps"] = 8
        elif (
            ModelName.FLUX_1_SCHNELL.value in self.model_id
            or ModelName.FLUX_1_DEV.value in self.model_id
        ):
            # FluxPipeline does not support negative prompt per diffusers documentation
            # https://github.com/huggingface/diffusers/blob/750bd7920622b3fe538d20035d3f03855c5d6621/src/diffusers/pipelines/flux/pipeline_flux.py#L537
            if "negative_prompt" in kwargs:
                kwargs.pop("negative_prompt")

        # Allow users to specify multiple (negative) prompts using the '|' separator.
        prompts = split_prompt(prompt, max_splits=3)
        prompt = prompts.pop("prompt")
        kwargs.update(prompts)
        neg_prompts = split_prompt(
            kwargs.pop("negative_prompt", ""),
            key_prefix="negative_prompt",
            max_splits=3,
        )
        kwargs.update(neg_prompts)

        try:
            outputs = self.ldm(prompt=prompt, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(outputs.images)
        else:
            has_nsfw_concept = [None] * len(outputs.images)

        return outputs.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"TextToImagePipeline model_id={self.model_id}"

from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import AnimateDiffPipeline, MotionAdapter, DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler
from huggingface_hub import file_download, hf_hub_download
from safetensors.torch import load_file
import torch
import PIL
from typing import List
import logging
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True

class TextToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True

        self.model_id = model_id

        if self.model_id == "ByteDance/AnimateDiff-Lightning":
            kwargs["torch_dtype"] = torch.float16
            adapter = MotionAdapter().to(torch_device, torch.float16)
            adapter.load_state_dict(load_file(hf_hub_download("ByteDance/AnimateDiff-Lightning", "animatediff_lightning_4step_diffusers.safetensors"), device="cuda"))
            kwargs["motion_adapter"] = adapter
            self.ldm = AnimateDiffPipeline.from_pretrained("digiplay/AbsoluteReality_v1.8.1", **kwargs)
            self.ldm.scheduler = EulerDiscreteScheduler.from_config(self.ldm.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
            self.ldm.to(torch_device)
        elif self.model_id == "ali-vilab/text-to-video-ms-1.7b":
            self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs)
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config)
            self.ldm.to(torch_device)
        else:
            self.ldm = DiffusionPipeline.from_pretrained(model_id, **kwargs).to(torch_device)
        self.ldm.enable_vae_slicing()

        if os.environ.get("SFAST"):
            logger.info(
                "TextToVideoPipeline will be dynamicallly compiled with stable-fast for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

    def __call__(self, prompt: str, **kwargs) -> List[List[PIL.Image]]:
        # ali-vilab/text-to-video-ms-1.7b has a limited parameter set
        if self.model_id == "ali-vilab/text-to-video-ms-1.7b":
            kwargs["num_inference_steps"] = 25
            if "fps" in kwargs:
                del kwargs["fps"]
            if "motion_bucket_id" in kwargs:
                del kwargs["motion_bucket_id"]
            if "noise_aug_strength" in kwargs:
                del kwargs["noise_aug_strength"]
        elif self.model_id == "ByteDance/AnimateDiff-Lightning":
            kwargs["num_inference_steps"] = 4
            kwargs["guidance_scale"] = 1
            if "fps" in kwargs:
                del kwargs["fps"]
            if "motion_bucket_id" in kwargs:
                del kwargs["motion_bucket_id"]
            if "noise_aug_strength" in kwargs:
                del kwargs["noise_aug_strength"]

        seed = kwargs.pop("seed", None)
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        return self.ldm(prompt, **kwargs).frames

    def __str__(self) -> str:
        return f"TextToVideoPipeline model_id={self.model_id}"

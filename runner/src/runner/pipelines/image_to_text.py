import logging
import os

import torch
from huggingface_hub import file_download
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from runner.pipelines.base import Pipeline
from runner.pipelines.utils import get_model_dir, get_torch_device
from runner.utils.errors import InferenceError

logger = logging.getLogger(__name__)


class ImageToTextPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {}

        self.torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if self.torch_device != "cpu" and has_fp16_variant:
            logger.info("ImageToTextPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("ImageToTextPipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        self.tm = BlipForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=get_model_dir(),
            **kwargs,
        ).to(self.torch_device)

        self.processor = BlipProcessor.from_pretrained(
            model_id, cache_dir=get_model_dir()
        )

    def __call__(self, prompt: str, image: Image, **kwargs) -> str:
        inputs = self.processor(image, prompt, return_tensors="pt").to(
            self.torch_device
        )
        out = self.tm.generate(**inputs)

        try:
            return self.processor.decode(out[0], skip_special_tokens=True)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

    def __str__(self) -> str:
        return f"ImageToTextPipeline model_id={self.model_id}"

    @property
    def router(self):
        from runner.routes import image_to_text
        return image_to_text.router

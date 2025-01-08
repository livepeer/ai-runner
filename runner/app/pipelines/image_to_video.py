import logging
import os
import time
from typing import List, Optional, Tuple

import PIL
import torch
from diffusers import LTXImageToVideoPipeline, StableVideoDiffusionPipeline
from huggingface_hub import file_download
from PIL import ImageFile

from app.pipelines.base import Pipeline
from app.pipelines.utils import SafetyChecker, get_model_dir, get_torch_device
from app.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SFAST_WARMUP_ITERATIONS = 2  # Model warm-up iterations when SFAST is enabled.


class ImageToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        self.pipeline_name = ""

        self.model_id = model_id
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
        if torch_device.type != "cpu" and has_fp16_variant:
            logger.info("ImageToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        try:
            if any(substring in model_id.lower() for substring in ("ltx-video", "ltx")):
                logger.info("Loading LTXImageToVideoPipeline for model_id: %s", model_id)
                self.pipeline_name = "LTXImageToVideoPipeline"
                self.ldm = LTXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
            else:
                logger.info("Loading StableVideoDiffusionPipeline for model_id: %s", model_id)
                self.pipeline_name = "StableVideoDiffusionPipeline"
                self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        except Exception as loading_error:
            logger.error("Failed to load %s : %s." %(self.pipeline_name,loading_error))
            # Trying to load the LTXImageToVideoPipeline if the StableVideoDiffusionPipeline fails to load and there is a chance that model name doesn't match the if condition for LTX-Video
            # (for future extra models support)
            try:
                logger.info("Trying LTXImageToVideoPipeline for model_id: %s", model_id)
                self.pipeline_name = "LTXImageToVideoPipeline"
                self.ldm = LTXImageToVideoPipeline.from_pretrained(model_id, **kwargs)
            except Exception as loading_error:
                logger.error("Failed to load both LTXImageToVideoPipeline and StableVideoDiffusionPipeline: %s. Please ensure the model ID is compatible.", loading_error)
                raise loading_error
            

        self.ldm.to(get_torch_device())

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        deepcache_enabled = os.getenv("DEEPCACHE", "").strip().lower() == "true"
        if sfast_enabled and deepcache_enabled:
            logger.warning(
                "Both 'SFAST' and 'DEEPCACHE' are enabled. This is not recommended "
                "as it may lead to suboptimal performance. Please disable one of them."
            )

        if sfast_enabled:
            logger.info(
                "ImageToVideoPipeline will be dynamically compiled with stable-fast "
                "for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # NOTE: Initial calls may be slow due to compilation. Subsequent calls will
            # be faster.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                # Retrieve default model params.
                # TODO: Retrieve defaults from Pydantic class in route.
                warmup_kwargs = {
                    "image": PIL.Image.new("RGB", (576, 1024)),
                    "height": 576,
                    "width": 1024,
                    "fps": 6,
                    "motion_bucket_id": 127,
                    "noise_aug_strength": 0.02,
                    "decode_chunk_size": 4,
                }

                logger.info("Warming up ImageToVideoPipeline pipeline...")
                total_time = 0
                for ii in range(SFAST_WARMUP_ITERATIONS):
                    t = time.time()
                    try:
                        self.ldm(**warmup_kwargs).frames
                    except Exception as e:
                        # FIXME: When out of memory, pipeline is corrupted.
                        logger.error(f"ImageToVideoPipeline warmup error: {e}")
                        raise e
                    iteration_time = time.time() - t
                    total_time += iteration_time
                    logger.info(
                        "Warmup iteration %s took %s seconds", ii + 1, iteration_time
                    )
                logger.info("Total warmup time: %s seconds", total_time)

        if deepcache_enabled:
            logger.info(
                "TextToImagePipeline will be optimized with DeepCache for %s",
                model_id,
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            self.ldm = enable_deepcache(self.ldm)

        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

    def __call__(
        self, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        safety_check = kwargs.pop("safety_check", True)

        if self.pipeline_name == "LTXImageToVideoPipeline":
            del kwargs["fps"]
            del kwargs["motion_bucket_id"]
            del kwargs["noise_aug_strength"]
        elif self.pipeline_name == "StableVideoDiffusionPipeline":
            del kwargs["prompt"]
            del kwargs["negative_prompt"]

        if "decode_chunk_size" not in kwargs:
            # Decrease decode_chunk_size to reduce memory usage.
            kwargs["decode_chunk_size"] = 4

        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        if "num_inference_steps" in kwargs and (
            kwargs["num_inference_steps"] is None or kwargs["num_inference_steps"] < 1
        ):
            del kwargs["num_inference_steps"]

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images([image])
        else:
            has_nsfw_concept = [None]

        try:
            outputs = self.ldm(image, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return outputs.frames, has_nsfw_concept

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"

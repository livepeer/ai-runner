"""This module contains several utility functions."""

import json
import logging
import os
import re
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from PIL import Image
from torch import dtype as TorchDtype
from transformers import CLIPImageProcessor

logger = logging.getLogger(__name__)

LORA_LIMIT = 4  # Max number of LoRas that can be requested at once.
LORA_MAX_LOADED = 12  # Number of LoRas to keep in memory.
LORA_FREE_VRAM_THRESHOLD = 2.0  # VRAM threshold (GB) to start evicting LoRas.


def get_model_dir() -> Path:
    return Path(os.environ["MODEL_DIR"])


def get_model_path(model_id: str) -> Path:
    return get_model_dir() / model_id.lower()


def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def validate_torch_device(device_name: str) -> bool:
    """Checks if the given PyTorch device name is valid and available.

    Args:
        device_name: Name of the device ('cuda:0', 'cuda', 'cpu').

    Returns:
        True if valid and available, False otherwise.
    """
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            # Check if CUDA is available and the specified index is within range
            if device.index is None:
                return torch.cuda.is_available()
            else:
                return device.index < torch.cuda.device_count()
        return True
    except RuntimeError:
        return False


def is_lightning_model(model_id: str) -> bool:
    """Checks if the model is a Lightning model.

    Args:
        model_id: Model ID.

    Returns:
        True if the model is a Lightning model, False otherwise.
    """
    return re.search(r"[-_]lightning", model_id, re.IGNORECASE) is not None


def is_turbo_model(model_id: str) -> bool:
    """Checks if the model is a Turbo model.

    Args:
        model_id: Model ID.

    Returns:
        True if the model is a Turbo model, False otherwise.
    """
    return re.search(r"[-_]turbo", model_id, re.IGNORECASE) is not None


def split_prompt(
    input_prompt: str,
    separator: str = "|",
    key_prefix: str = "prompt",
    max_splits: int = -1,
) -> Dict[str, str]:
    """Splits an input prompt into prompts, including the main prompt, with customizable
    key naming.

    Args:
        input_prompt (str): The input prompt string to be split.
        separator (str): The character used to split the input prompt. Defaults to '|'.
        key_prefix (str): Prefix for keys in the returned dictionary for all prompts,
            including the main prompt. Defaults to 'prompt'.
        max_splits (int): Maximum number of splits to perform. Defaults to -1 (no
            limit).

    Returns:
        Dict[str, str]: A dictionary of all prompts, including the main prompt.
    """
    prompts = [
        prompt.strip()
        for prompt in input_prompt.split(separator, max_splits)
        if prompt.strip()
    ]
    if not prompts:
        return {}

    start_index = max(1, len(prompts) - max_splits) if max_splits >= 0 else 1

    prompt_dict = {f"{key_prefix}": prompts[0]}
    prompt_dict.update(
        {
            f"{key_prefix}_{i+1}": prompt
            for i, prompt in enumerate(prompts[1:], start=start_index)
        }
    )

    return prompt_dict


class SafetyChecker:
    """Checks images for unsafe or inappropriate content using a pretrained model.

    Attributes:
        device (str): Device for inference.
    """

    def __init__(
        self,
        device: Optional[str] = "cuda",
        dtype: Optional[TorchDtype] = torch.float16,
    ):
        """Initializes the SafetyChecker.

        Args:
            device: Device for inference. Defaults to "cuda".
            dtype: Data type for inference. Defaults to `torch.float16`.
        """
        device = device.lower() if device else device
        if not validate_torch_device(device):
            default_device = get_torch_device()
            logger.warning(
                f"Device '{device}' not found. Defaulting to '{default_device}'."
            )
            device = default_device

        self.device = device
        self._dtype = dtype
        
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device)
        self._feature_extractor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def check_nsfw_images(
        self, images: list[Image.Image]
    ) -> tuple[list[Image.Image], list[bool]]:
        """Checks images for unsafe content.

        Args:
            images: Images to check.

        Returns:
            Tuple of images and corresponding NSFW flags.
        """
        safety_checker_input = self._feature_extractor(images, return_tensors="pt").to(
            self.device
        )
        images_np = [np.array(img) for img in images]
        _, has_nsfw_concept = self._safety_checker(
            images=images_np,
            clip_input=safety_checker_input.pixel_values.to(self._dtype),
        )
        return images, has_nsfw_concept


def is_numeric(val: Any) -> bool:
    """Check if the given value is numeric.

    Args:
        s: Value to check.

    Returns:
        True if the value is numeric, False otherwise.
    """
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


class LoraLoadingError(Exception):
    """Exception raised for errors during LoRa loading."""

    def __init__(self, message="Error loading LoRas", original_exception=None):
        """Initialize the exception.
        Args:
            message: The error message.
            original_exception: The original exception that caused the error.
        """
        if original_exception:
            message = f"{message}: {original_exception}"
        super().__init__(message)
        self.original_exception = original_exception


class LoraLoader:
    """Utility class to load LoRas and set their weights into a given pipeline.

    Attributes:
        pipeline: Diffusion pipeline on which the LoRas are loaded.
        loras_enabled: Flag to enable or disable LoRas.
    """

    def __init__(self, pipeline: DiffusionPipeline):
        """Initializes the LoraLoader.

        Args:
            pipeline: Diffusion pipeline to load LoRas into.
        """
        self.pipeline = pipeline
        self.loras_enabled = False

    def _get_loaded_loras(self) -> List[str]:
        """Returns the names of the loaded LoRas.

        Returns:
            List of loaded LoRa names.
        """
        loaded_loras_dict = self.pipeline.get_list_adapters()
        seen = set()
        return [
            lora
            for loras in loaded_loras_dict.values()
            for lora in loras
            if lora not in seen and not seen.add(lora)
        ]

    def _evict_loras_if_needed(self, request_loras: dict) -> None:
        """Evict the oldest unused LoRa until free memory is above the threshold or the
        number of loaded LoRas is below the maximum allowed.

        Args:
            request_loras: list of requested LoRas.
        """
        while True:
            free_memory_gb = (
                torch.cuda.mem_get_info(device=self.pipeline.device)[0] / 1024**3
            )
            loaded_loras = self._get_loaded_loras()
            memory_limit_reached = free_memory_gb < LORA_FREE_VRAM_THRESHOLD

            # Break if memory is sufficient, LoRas within limit, or no LoRas to evict.
            if (
                not memory_limit_reached
                and len(loaded_loras) < LORA_MAX_LOADED
                or not any(lora not in request_loras for lora in loaded_loras)
            ):
                break

            # Evict the oldest unused LoRa.
            for lora in loaded_loras:
                if lora not in request_loras:
                    self.pipeline.delete_adapters(lora)
                    break
        if memory_limit_reached:
            torch.cuda.empty_cache()

    def load_loras(self, loras_json: str) -> None:
        """Loads LoRas and sets their weights into the pipeline managed by this
        LoraLoader.

        Args:
            loras_json: A JSON string containing key-value pairs, where the key is the
                repository to load LoRas from and the value is the strength (a float
                with a minimum value of 0.0) to assign to the LoRa.

        Raises:
            LoraLoadingError: If an error occurs during LoRa loading.
        """
        try:
            lora_dict = json.loads(loras_json)
        except json.JSONDecodeError:
            error_message = f"Unable to parse '{loras_json}' as JSON."
            logger.warning(error_message)
            raise LoraLoadingError(error_message)

        # Parse Lora strengths and check for invalid values.
        invalid_loras = {
            adapter: val
            for adapter, val in lora_dict.items()
            if not is_numeric(val) or float(val) < 0.0
        }
        if invalid_loras:
            error_message = (
                "All strengths must be numbers greater than or equal to 0.0."
            )
            logger.warning(error_message)
            raise LoraLoadingError(error_message)
        lora_dict = {adapter: float(val) for adapter, val in lora_dict.items()}

        # Disable LoRas if none are provided.
        if not lora_dict:
            self.disable_loras()
            return

        # Limit the number of active loras to prevent pipeline slowdown.
        if len(lora_dict) > LORA_LIMIT:
            raise LoraLoadingError(f"Too many LoRas provided. Maximum is {LORA_LIMIT}.")

        # Re-enable LoRas if they were disabled.
        self.enable_loras()

        # Load new LoRa adapters.
        loaded_loras = self._get_loaded_loras()
        try:
            for adapter in lora_dict.keys():
                # Load new Lora weights and evict the oldest unused Lora if necessary.
                if adapter not in loaded_loras:
                    self.pipeline.load_lora_weights(adapter, adapter_name=adapter)
                    self._evict_loras_if_needed(list(lora_dict.keys()))
        except Exception as e:
            # Delete failed adapter and log the error.
            self.pipeline.delete_adapters(adapter)
            torch.cuda.empty_cache()
            if "not found in the base model" in str(e):
                error_message = (
                    "LoRa incompatible with base model: "
                    f"'{self.pipeline.name_or_path}'"
                )
            elif getattr(e, "server_message", "") == "Repository not found":
                error_message = f"LoRa repository '{adapter}' not found"
            else:
                error_message = f"Unable to load LoRas for adapter '{adapter}'"
                logger.exception(e)
            raise LoraLoadingError(error_message)

        # Set unused LoRas strengths to 0.0.
        for lora in loaded_loras:
            if lora not in lora_dict:
                lora_dict[lora] = 0.0

        # Set the lora adapter strengths.
        self.pipeline.set_adapters(*map(list, zip(*lora_dict.items())))

    def disable_loras(self) -> None:
        """Disables all LoRas in the pipeline."""
        if self.loras_enabled:
            self.pipeline.disable_lora()
            self.loras_enabled = False

    def enable_loras(self) -> None:
        """Enables all LoRas in the pipeline."""
        if not self.loras_enabled:
            self.pipeline.enable_lora()
            self.loras_enabled = True


class MemoryInfo:
    def __init__(self, gpu_memory, cpu_memory, num_gpus):
        self.gpu_memory = gpu_memory
        self.cpu_memory = cpu_memory
        self.num_gpus = num_gpus

    def __repr__(self):
        return f"<MemoryInfo: GPUs={self.num_gpus}, CPU Memory={self.cpu_memory}, GPU Memory={self.gpu_memory}>"


def get_max_memory() -> MemoryInfo:
    num_gpus = torch.cuda.device_count()
    gpu_memory = {
        i: f"{torch.cuda.get_device_properties(i).total_memory // 1024**3}GiB" for i in range(num_gpus)}
    cpu_memory = f"{psutil.virtual_memory().available // 1024**3}GiB"

    memory_info = MemoryInfo(gpu_memory=gpu_memory,
                             cpu_memory=cpu_memory, num_gpus=num_gpus)

    return memory_info

import logging
from typing import List, Optional, Tuple

import torch
import PIL
from PIL import ImageFile
from sam2.sam2_image_predictor import SAM2ImagePredictor

from runner.pipelines.base import Pipeline
from runner.pipelines.utils import get_model_dir, get_torch_device
from runner.utils.errors import InferenceError

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class SegmentAnything2Pipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()

        self.tm = SAM2ImagePredictor.from_pretrained(
            model_id=model_id,
            device=torch_device,
            **kwargs,
        )

    def __call__(
        self, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        try:
            self.tm.set_image(image)
            prediction = self.tm.predict(**kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise e
        except Exception as e:
            raise InferenceError(original_exception=e)

        return prediction

    def __str__(self) -> str:
        return f"SegmentAnything2Pipeline model_id={self.model_id}"

    @property
    def router(self):
        from runner.routes import segment_anything_2
        return segment_anything_2.router

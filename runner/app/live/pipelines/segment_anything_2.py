import io
import logging
import threading
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from Sam2Wrapper import Sam2Wrapper
from .interface import Pipeline

logger = logging.getLogger(__name__)

class Sam2LiveParams(BaseModel):
    class Config:
        extra = "forbid"

    model_id: str = "facebook/sam2-hiera-tiny"
    point_coords: List[List[int]] = [[1, 1]]
    point_labels: List[int] = [1]
    show_point: bool = False

    def __init__(self, **data): 
        super().__init__(**data)

class Sam2Live(Pipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe: Optional[Sam2Wrapper] = None
        self.first_frame = True
        self.update_params(**params)

    def update_params(self, **params):
        new_params = Sam2LiveParams(**params)
        self.params = new_params

        logging.info(f"Setting parameters for sam2")
        self.pipe = Sam2Wrapper(
            model_id_or_path=self.params.model_id,
            point_coords=self.params.point_coords,
            point_labels=self.params.point_labels,
            show_point=self.params.point_labels,
            # Add additional as needed
        )

        self.params = new_params
        self.first_frame = True

    def _process_mask(self, mask: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """Process and resize mask if needed."""
        if mask.shape[0] == 0:
            return np.zeros((frame_shape[0], frame_shape[1]), dtype="uint8")
            
        mask = (mask[0, 0] > 0).cpu().numpy().astype("uint8") * 255
        if mask.shape[:2] != frame_shape[:2]:
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
        return mask

    def process_frame(self, frame: Image.Image, **params) -> Image.Image:
        frame_lock = threading.Lock()
        start_time = time.time()
        try:
            if params:
                self.update_params(**params)

            # Convert image formats
            t0 = time.time()
            frame_array = np.array(frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
            logger.debug(f"Image conversion took {(time.time() - t0)*1000:.2f}ms")

            if self.first_frame:
                t0 = time.time()
                self.pipe.predictor.load_first_frame(frame)
                
                for idx, point in enumerate(self.params.point_coords):
                    _, _, mask_logits = self.pipe.predictor.add_new_prompt(
                        frame_idx=0, 
                        obj_id=idx + 1, 
                        points=[point], 
                        labels=[self.params.point_labels[idx]]
                    )
                logger.debug(f"First frame processing took {(time.time() - t0)*1000:.2f}ms")
                self.first_frame = False
            else:
                t0 = time.time()
                _, mask_logits = self.pipe.predictor.track(frame)
                logger.debug(f"Frame tracking took {(time.time() - t0)*1000:.2f}ms")

            # Process mask and create overlay
            t0 = time.time()
            mask = self._process_mask(mask_logits, frame_bgr.shape)
            logger.debug(f"Mask processing took {(time.time() - t0)*1000:.2f}ms")

            # Create an overlay by combining the original frame and the mask
            colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            colored_mask[mask > 0] = [255, 0, 255]  # BGR format: # Add a purple tint to the mask
            overlay = cv2.addWeighted(frame_bgr, 1, colored_mask, 1, 0)
            
            # Draw points on the overlay
            if hasattr(self.params, 'show_point') and self.params.show_point:
                for point in self.params.point_coords:
                    cv2.circle(overlay, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot

            # Convert back to PIL Image
            t0 = time.time()
            with frame_lock:
                _, buffer = cv2.imencode('.jpg', overlay)
                result = Image.open(io.BytesIO(buffer.tobytes()))
            logger.debug(f"Final conversion took {(time.time() - t0)*1000:.2f}ms")

            total_time = time.time() - start_time
            logger.debug(f"Total frame processing time: {total_time*1000:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return Image.new("RGB", frame.size, (255, 255, 255))
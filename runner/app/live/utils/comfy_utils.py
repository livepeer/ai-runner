import logging
import json
from typing import Tuple, Dict
from trickle import DEFAULT_WIDTH, DEFAULT_HEIGHT

class ComfyUtils:
    @staticmethod
    def update_latent_image_dimensions(workflow: dict | str | None, width: int, height: int) -> dict:
        """Update the EmptyLatentImage node dimensions in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary or string
            width: New width to set
            height: New height to set
            
        Returns:
            Updated workflow dictionary with new dimensions
        """

        if workflow is None:
            return {}
        try:
            if isinstance(workflow, str):
                workflow = json.loads(workflow)
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse workflow JSON: {e}")
            return {}
            
        try:
            for node_id, node in workflow.items():
                if node.get("class_type") == "EmptyLatentImage":
                    if "inputs" not in node:
                        node["inputs"] = {}
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height
                    break
        except Exception as e:
            logging.warning(f"Failed to update dimensions in workflow: {e}")
            
        return workflow

    @staticmethod
    def get_latent_image_dimensions(workflow: dict | str | None) -> tuple[int, int]:
        """Get dimensions from the EmptyLatentImage node in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary
            
        Returns:
            Tuple of (width, height) from the latent image. Returns default dimensions if not found or on error.
        """

        if workflow is None:
            return DEFAULT_WIDTH, DEFAULT_HEIGHT
        
        if isinstance(workflow, str):
            workflow = json.loads(workflow)
        
        try:
            for node_id, node in workflow.items():
                if node.get("class_type") == "EmptyLatentImage":
                    inputs = node.get("inputs", {})
                    width = inputs.get("width")
                    height = inputs.get("height")
                    if width is not None and height is not None:
                        return width, height
                    logging.warning("Incomplete dimensions in latent image node")
                    break
        except Exception as e:
            logging.warning(f"Failed to extract dimensions from workflow: {e}")
        
        # Return defaults if dimensions not found or on any error
        logging.info(f"Using default dimensions {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
        return DEFAULT_WIDTH, DEFAULT_HEIGHT

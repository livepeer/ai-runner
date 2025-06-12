import logging
import json
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
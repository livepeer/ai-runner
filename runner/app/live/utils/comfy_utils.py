import logging

class ComfyUtils:
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    
    @staticmethod
    def get_latent_image_dimensions(workflow: dict | None) -> tuple[int, int]:
        """Get dimensions from the EmptyLatentImage node in the workflow.
        
        Args:
            workflow: The workflow JSON dictionary
            
        Returns:
            Tuple of (width, height) from the latent image. Returns default dimensions if not found or on error.
        """

        if workflow is None:
            return ComfyUtils.DEFAULT_WIDTH, ComfyUtils.DEFAULT_HEIGHT
        
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
        logging.info(f"Using default dimensions {ComfyUtils.DEFAULT_WIDTH}x{ComfyUtils.DEFAULT_HEIGHT}")
        return ComfyUtils.DEFAULT_WIDTH, ComfyUtils.DEFAULT_HEIGHT 
from .interface import Pipeline

def load_pipeline(name: str) -> Pipeline:
    if name == "comfyui":
        from .comfyui import ComfyUI
        return ComfyUI()
    elif name == "noop":
        from .noop import Noop
        return Noop()
    raise ValueError(f"Unknown pipeline: {name}")

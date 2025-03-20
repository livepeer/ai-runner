from .interface import Pipeline

def load_pipeline(name: str, **params) -> Pipeline:
    if name == "comfyui":
        from .comfyui import ComfyUI
        return ComfyUI(**params)
    elif name == "noop":
        from .noop import Noop
        return Noop(**params)
    raise ValueError(f"Unknown pipeline: {name}")

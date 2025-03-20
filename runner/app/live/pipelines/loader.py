from .interface import Pipeline

def load_pipeline(name: str, **params) -> Pipeline:
    if name == "streamdiffusion":
        from .streamdiffusion import StreamDiffusion
        return StreamDiffusion(**params)
    elif name == "comfyui":
        from .comfyui import ComfyUI
        return ComfyUI(**params)
    elif name == "noop":
        from .noop import Noop
        return Noop(**params)
    elif name == "segment_anything_2":
        from .segment_anything_2 import Sam2Live
        return Sam2Live(**params)
    raise ValueError(f"Unknown pipeline: {name}")

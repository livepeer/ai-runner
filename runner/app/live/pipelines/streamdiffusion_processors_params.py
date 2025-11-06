from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar, Dict, Optional, Tuple, Type, Literal, get_args, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseProcessorParams(BaseModel):
    """Common parameter fields shared by StreamDiffusion preprocessors."""

    model_config = ConfigDict(extra="forbid")

    requires_pipeline_ref: ClassVar[bool] = False

    image_resolution: Optional[int] = Field(
        default=None,
        ge=1,
        description="Square resolution fallback when explicit width/height are not provided.",
    )
    image_width: Optional[int] = Field(default=None, ge=1)
    image_height: Optional[int] = Field(default=None, ge=1)
    device: Optional[str] = None
    dtype: Optional[str] = None

    def runtime_kwargs(self) -> Dict[str, object]:
        """Utility to produce kwargs for processor constructors."""
        return self.model_dump(exclude_none=True)


class CannyParams(BaseProcessorParams):
    low_threshold: int = Field(default=100, ge=0, le=255)
    high_threshold: int = Field(default=200, ge=0, le=255)


class DepthParams(BaseProcessorParams):
    model_name: str = Field(default="Intel/dpt-large")
    detect_resolution: int = Field(default=512, ge=64)
    image_resolution: int = Field(default=512, ge=64)


class DepthAnythingTensorRTParams(BaseProcessorParams):
    engine_path: Optional[str] = None
    detect_resolution: int = Field(default=518, ge=64)
    image_resolution: int = Field(default=512, ge=64)


class OpenPoseParams(BaseProcessorParams):
    detect_resolution: int = Field(default=512, ge=64)
    image_resolution: int = Field(default=512, ge=64)
    include_hands: bool = False
    include_face: bool = False


class LineartParams(BaseProcessorParams):
    detect_resolution: int = Field(default=512, ge=64)
    image_resolution: int = Field(default=512, ge=64)
    coarse: bool = True
    anime_style: bool = False


class StandardLineartParams(BaseProcessorParams):
    detect_resolution: int = Field(default=512, ge=64)
    image_resolution: int = Field(default=512, ge=64)
    gaussian_sigma: float = Field(default=6.0, gt=0.0)
    intensity_threshold: int = Field(default=8, ge=0)


class PassthroughParams(BaseProcessorParams):
    image_resolution: int = Field(default=512, ge=64)


class ExternalParams(BaseProcessorParams):
    image_resolution: int = Field(default=512, ge=64)
    validate_input: bool = True


class SoftEdgeParams(BaseProcessorParams):
    """Soft edge detector has no additional tunables beyond the base fields."""


class HEDParams(BaseProcessorParams):
    safe: bool = True


class FeedbackParams(BaseProcessorParams):
    requires_pipeline_ref: ClassVar[bool] = True

    image_resolution: int = Field(default=512, ge=64)
    feedback_strength: float = Field(default=0.5, ge=0.0, le=1.0)


class LatentFeedbackParams(BaseProcessorParams):
    requires_pipeline_ref: ClassVar[bool] = True

    feedback_strength: float = Field(default=0.5, ge=0.0, le=1.0)


class SharpenParams(BaseProcessorParams):
    sharpen_intensity: float = Field(default=1.5, gt=0.0)
    unsharp_radius: float = Field(default=1.0, gt=0.0)
    edge_enhancement: float = Field(default=0.5, ge=0.0, le=2.0)
    detail_boost: float = Field(default=0.3, ge=0.0, le=1.0)
    noise_reduction: float = Field(default=0.1, ge=0.0, le=0.5)
    multi_scale: bool = True


class BlurParams(BaseProcessorParams):
    blur_intensity: float = Field(default=2.0, gt=0.0)
    kernel_size: int = Field(default=15, ge=3)

    @field_validator("kernel_size")
    @classmethod
    def ensure_odd_kernel(cls, value: int) -> int:
        """Match runtime behaviour by nudging even kernels to the next odd value."""
        return value if value % 2 == 1 else value + 1


class UpscaleParams(BaseProcessorParams):
    scale_factor: float = Field(default=2.0, ge=1.0)
    algorithm: Literal["bilinear", "lanczos", "bicubic", "nearest"] = "bilinear"


class RealESRGANParams(BaseProcessorParams):
    enable_tensorrt: bool = True
    force_rebuild: bool = False


class MediaPipePoseParams(BaseProcessorParams):
    detect_resolution: int = Field(default=256, ge=64)
    image_resolution: int = Field(default=512, ge=64)
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model_complexity: int = Field(default=1, ge=0, le=2)
    static_image_mode: bool = False
    draw_hands: bool = True
    draw_face: bool = False
    line_thickness: int = Field(default=2, ge=1)
    circle_radius: int = Field(default=4, ge=1)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_smoothing: bool = True
    smoothing_factor: float = Field(default=0.7, ge=0.0, le=1.0)


class MediaPipeSegmentationParams(BaseProcessorParams):
    detect_resolution: int = Field(default=512, ge=64)
    image_resolution: int = Field(default=512, ge=64)
    model_selection: int = Field(default=1, ge=0, le=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    blur_radius: int = Field(default=0, ge=0)
    invert_mask: bool = False
    output_mode: Literal["binary", "alpha", "background"] = "binary"
    background_color: Tuple[int, int, int] = Field(default=(0, 0, 0))

    @field_validator("background_color")
    @classmethod
    def clamp_background_color(cls, value: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError("background_color must contain three RGB values")
        r, g, b = value
        for channel in (r, g, b):
            if not 0 <= channel <= 255:
                raise ValueError("background_color components must be between 0 and 255")
        return value


class PoseTensorRTParams(BaseProcessorParams):
    engine_path: Optional[str] = None
    detect_resolution: int = Field(default=640, ge=64)
    image_resolution: int = Field(default=512, ge=64)


class TemporalNetTensorRTParams(BaseProcessorParams):
    requires_pipeline_ref: ClassVar[bool] = True

    image_resolution: int = Field(default=512, ge=64)
    flow_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    detect_resolution: int = Field(default=512, ge=64)
    output_format: Literal["concat", "warped_only"] = "concat"
    enable_tensorrt: bool = True
    force_rebuild: bool = False


ImageProcessorLiteral = Literal[
    "blur",
    "canny",
    "depth",
    "depth_tensorrt",
    "external",
    "feedback",
    "hed",
    "lineart",
    "mediapipe_pose",
    "mediapipe_segmentation",
    "openpose",
    "passthrough",
    "pose_tensorrt",
    "realesrgan_trt",
    "sharpen",
    "soft_edge",
    "standard_lineart",
    "temporal_net_tensorrt",
    "upscale",
]

LatentProcessorLiteral = Literal["latent_feedback"]

ControlNetPreprocessorLiteral = Literal[
    "canny",
    "depth",
    "depth_tensorrt",
    "external",
    "feedback",
    "hed",
    "lineart",
    "mediapipe_pose",
    "mediapipe_segmentation",
    "openpose",
    "passthrough",
    "pose_tensorrt",
    "soft_edge",
    "standard_lineart",
    "temporal_net_tensorrt",
]

IMAGE_PROCESSORS: Tuple[str, ...] = cast(Tuple[str, ...], get_args(ImageProcessorLiteral))

LATENT_PROCESSORS: Tuple[str, ...] = cast(Tuple[str, ...], get_args(LatentProcessorLiteral))

# ControlNet-compatible subset keeps parity with StreamDiffusion expectations.
CONTROLNET_PREPROCESSORS: Tuple[str, ...] = cast(
    Tuple[str, ...],
    get_args(ControlNetPreprocessorLiteral),
)


# Registries make it easy to construct configs when only the literal name is known.
IMAGE_PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessorParams]] = {
    "blur": BlurParams,
    "canny": CannyParams,
    "depth": DepthParams,
    "depth_tensorrt": DepthAnythingTensorRTParams,
    "external": ExternalParams,
    "feedback": FeedbackParams,
    "hed": HEDParams,
    "lineart": LineartParams,
    "mediapipe_pose": MediaPipePoseParams,
    "mediapipe_segmentation": MediaPipeSegmentationParams,
    "openpose": OpenPoseParams,
    "passthrough": PassthroughParams,
    "pose_tensorrt": PoseTensorRTParams,
    "realesrgan_trt": RealESRGANParams,
    "sharpen": SharpenParams,
    "soft_edge": SoftEdgeParams,
    "standard_lineart": StandardLineartParams,
    "temporal_net_tensorrt": TemporalNetTensorRTParams,
    "upscale": UpscaleParams,
}

LATENT_PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessorParams]] = {
    "latent_feedback": LatentFeedbackParams,
}

CONTROLNET_PREPROCESSOR_REGISTRY: Dict[str, Type[BaseProcessorParams]] = {
    key: IMAGE_PROCESSOR_REGISTRY[key] for key in CONTROLNET_PREPROCESSORS
}

# IPAdapter embedding processors are intentionally omitted – they belong to the IPAdapter
# subsystem, not to the generic image/latent hook surface.


preprocessors = SimpleNamespace(
    Base=BaseProcessorParams,
    Blur=BlurParams,
    Canny=CannyParams,
    Depth=DepthParams,
    DepthTensorRT=DepthAnythingTensorRTParams,
    External=ExternalParams,
    Feedback=FeedbackParams,
    HED=HEDParams,
    LatentFeedback=LatentFeedbackParams,
    Lineart=LineartParams,
    MediaPipePose=MediaPipePoseParams,
    MediaPipeSegmentation=MediaPipeSegmentationParams,
    OpenPose=OpenPoseParams,
    Passthrough=PassthroughParams,
    PoseTensorRT=PoseTensorRTParams,
    RealESRGAN=RealESRGANParams,
    Sharpen=SharpenParams,
    SoftEdge=SoftEdgeParams,
    StandardLineart=StandardLineartParams,
    TemporalNetTensorRT=TemporalNetTensorRTParams,
    Upscale=UpscaleParams,
    IMAGE_REGISTRY=IMAGE_PROCESSOR_REGISTRY,
    LATENT_REGISTRY=LATENT_PROCESSOR_REGISTRY,
    CONTROLNET_REGISTRY=CONTROLNET_PREPROCESSOR_REGISTRY,
    IMAGE_LITERALS=IMAGE_PROCESSORS,
    LATENT_LITERALS=LATENT_PROCESSORS,
    CONTROLNET_LITERALS=CONTROLNET_PREPROCESSORS,
    IMAGE_LITERAL_TYPE=ImageProcessorLiteral,
    LATENT_LITERAL_TYPE=LatentProcessorLiteral,
    CONTROLNET_LITERAL_TYPE=ControlNetPreprocessorLiteral,
)


__all__ = [
    "BaseProcessorParams",
    "BlurParams",
    "CannyParams",
    "DepthParams",
    "DepthAnythingTensorRTParams",
    "ExternalParams",
    "FeedbackParams",
    "HEDParams",
    "LatentFeedbackParams",
    "LineartParams",
    "MediaPipePoseParams",
    "MediaPipeSegmentationParams",
    "OpenPoseParams",
    "PassthroughParams",
    "PoseTensorRTParams",
    "RealESRGANParams",
    "SharpenParams",
    "SoftEdgeParams",
    "StandardLineartParams",
    "TemporalNetTensorRTParams",
    "UpscaleParams",
    "ImageProcessorLiteral",
    "LatentProcessorLiteral",
    "ControlNetPreprocessorLiteral",
    "IMAGE_PROCESSORS",
    "LATENT_PROCESSORS",
    "CONTROLNET_PREPROCESSORS",
    "IMAGE_PROCESSOR_REGISTRY",
    "LATENT_PROCESSOR_REGISTRY",
    "CONTROLNET_PREPROCESSOR_REGISTRY",
    "preprocessors",
]



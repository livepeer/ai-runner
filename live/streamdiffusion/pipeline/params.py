from typing import Dict, List, Literal, Optional, Any, Tuple, TypeVar, Generic

from pydantic import BaseModel, model_validator, Field

from app.live.pipelines import BaseParams

ModelType = Literal["sd15", "sd21", "sdxl"]

# Module-level flag to skip ControlNet limit check during TensorRT compilation
_is_building_tensorrt_engines = False

IPADAPTER_SUPPORTED_TYPES: List[ModelType] = ["sd15", "sdxl"]

CONTROLNETS_BY_TYPE: Dict[ModelType, List[str]] = {
    "sd21": [
        "thibaud/controlnet-sd21-openpose-diffusers",
        "thibaud/controlnet-sd21-hed-diffusers",
        "thibaud/controlnet-sd21-canny-diffusers",
        "thibaud/controlnet-sd21-depth-diffusers",
        "thibaud/controlnet-sd21-color-diffusers",
        "daydreamlive/TemporalNet2-stable-diffusion-2-1",
    ],
    "sd15": [
        "lllyasviel/control_v11f1p_sd15_depth",
        "lllyasviel/control_v11f1e_sd15_tile",
        "lllyasviel/control_v11p_sd15_canny",
        "daydreamlive/TemporalNet2-stable-diffusion-v1-5",
    ],
    "sdxl": [
        "xinsir/controlnet-depth-sdxl-1.0",
        "xinsir/controlnet-canny-sdxl-1.0",
        "xinsir/controlnet-tile-sdxl-1.0",
        "daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0",
    ],
}

LCM_LORAS_BY_TYPE: Dict[ModelType, str] = {
    "sdxl": "latent-consistency/lcm-lora-sdxl",
    "sd15": "latent-consistency/lcm-lora-sdv1-5",
}

CACHED_ATTENTION_MIN_FRAMES = 1
CACHED_ATTENTION_MAX_FRAMES = 4

MODEL_ID_TO_TYPE: Dict[str, ModelType] = {
    "stabilityai/sd-turbo": "sd21",
    "stabilityai/sdxl-turbo": "sdxl",
    "prompthero/openjourney-v4": "sd15",
    "Lykon/dreamshaper-8": "sd15",
}

def get_model_type(model_id: str) -> ModelType:
    if model_id not in MODEL_ID_TO_TYPE:
        raise ValueError(f"Invalid model_id: {model_id}")
    return MODEL_ID_TO_TYPE[model_id]

ImageProcessorName = Literal[
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

LatentProcessorsName = Literal["latent_feedback"]

ProcessorParams = Dict[str, Any]

ProcessorTypeT = TypeVar("ProcessorTypeT", bound=str)

class SingleProcessorConfig(BaseModel, Generic[ProcessorTypeT]):
    class Config:
        extra = "forbid"

    type: ProcessorTypeT
    enabled: bool = True
    params: ProcessorParams = Field(default_factory=dict)

class ProcessingConfig(BaseModel, Generic[ProcessorTypeT]):
    class Config:
        extra = "forbid"

    enabled: bool = True
    processors: List[SingleProcessorConfig[ProcessorTypeT]] = Field(default_factory=list)

class ControlNetConfig(BaseModel):
    class Config:
        extra = "forbid"

    model_id: Literal[
        "thibaud/controlnet-sd21-openpose-diffusers",
        "thibaud/controlnet-sd21-hed-diffusers",
        "thibaud/controlnet-sd21-canny-diffusers",
        "thibaud/controlnet-sd21-depth-diffusers",
        "thibaud/controlnet-sd21-color-diffusers",
        "daydreamlive/TemporalNet2-stable-diffusion-2-1",
        "lllyasviel/control_v11f1p_sd15_depth",
        "lllyasviel/control_v11f1e_sd15_tile",
        "lllyasviel/control_v11p_sd15_canny",
        "daydreamlive/TemporalNet2-stable-diffusion-v1-5",
        "xinsir/controlnet-depth-sdxl-1.0",
        "xinsir/controlnet-canny-sdxl-1.0",
        "xinsir/controlnet-tile-sdxl-1.0",
        "daydreamlive/TemporalNet2-stable-diffusion-xl-base-1.0",
    ]

    conditioning_scale: float = 1.0
    conditioning_channels: int | None = Field(default=None, ge=1, le=6)
    preprocessor: ImageProcessorName = "passthrough"
    preprocessor_params: ProcessorParams = Field(default_factory=dict)
    enabled: bool = True
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0

class IPAdapterConfig(BaseModel):
    class Config:
        extra = "forbid"

    type: Literal["regular", "faceid"] = "regular"
    ipadapter_model_path: Optional[Literal[
        "h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
        "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin",
        "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
    ]] = None
    image_encoder_path: Optional[Literal[
        "h94/IP-Adapter/models/image_encoder",
        "h94/IP-Adapter/sdxl_models/image_encoder",
    ]] = None
    insightface_model_name: Optional[Literal["buffalo_l"]] = "buffalo_l"
    scale: float = 1.0
    weight_type: Optional[Literal[
        "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
        "weak input", "weak output", "weak middle", "strong middle",
        "style transfer", "composition", "strong style transfer",
        "style and composition", "style transfer precise", "composition precise"
    ]] = "linear"
    enabled: bool = True


class CachedAttentionConfig(BaseModel):
    class Config:
        extra = "forbid"

    enabled: bool = True
    max_frames: int = Field(
        default=1,
        ge=CACHED_ATTENTION_MIN_FRAMES,
        le=CACHED_ATTENTION_MAX_FRAMES,
    )
    interval: int = Field(default=1, ge=1, le=1440)


class StreamDiffusionParams(BaseParams):
    class Config:
        extra = "forbid"

    model_id: Literal[
        "stabilityai/sd-turbo",
        "stabilityai/sdxl-turbo",
        "prompthero/openjourney-v4",
        "Lykon/dreamshaper-8",
    ] = "stabilityai/sd-turbo"

    prompt: str | List[Tuple[str, float]] = "flowers"
    prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    normalize_prompt_weights: bool = True
    negative_prompt: str = "blurry, low quality, flat, 2d"
    guidance_scale: float = 1.0
    delta: float = 0.7
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    t_index_list: List[int] = Field(default_factory=lambda: [12, 20, 32])

    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"

    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"

    use_safety_checker: bool = True
    safety_checker_threshold: float = 0.5
    use_denoising_batch: bool = True
    do_add_noise: bool = True
    skip_diffusion: bool = False

    seed: int | List[Tuple[int, float]] = 789
    seed_interpolation_method: Literal["linear", "slerp"] = "linear"
    normalize_seed_weights: bool = True

    enable_similar_image_filter: bool = False
    similar_image_filter_threshold: float = 0.98
    similar_image_filter_max_skip_frame: int = 10

    controlnets: List[ControlNetConfig] = Field(default_factory=list)

    ip_adapter: IPAdapterConfig = Field(default_factory=lambda: IPAdapterConfig(enabled=False))
    ip_adapter_style_image_url: str = "https://storage.googleapis.com/lp-ai-assets/ipadapter_style_imgs/textures/vortex.jpeg"

    image_preprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    image_postprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    latent_preprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None
    latent_postprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None

    cached_attention: CachedAttentionConfig = Field(default_factory=lambda: CachedAttentionConfig(enabled=True))

    def get_output_resolution(self) -> tuple[int, int]:
        output_width, output_height = self.width, self.height

        if self.image_postprocessing and self.image_postprocessing.enabled:
            for proc in self.image_postprocessing.processors:
                if proc.enabled and proc.type in ["upscale", "realesrgan_trt"]:
                    scale_factor = 2.0 if proc.type == "realesrgan_trt" else proc.params.get("scale_factor", 2.0)
                    output_width = int(output_width * scale_factor)
                    output_height = int(output_height * scale_factor)

        return (output_width, output_height)

    @model_validator(mode="after")
    @staticmethod
    def check_t_index_list(model: "StreamDiffusionParams") -> "StreamDiffusionParams":
        if not (1 <= len(model.t_index_list) <= 4):
            raise ValueError("t_index_list must have between 1 and 4 elements")

        for i, value in enumerate(model.t_index_list):
            if not (0 <= value <= model.num_inference_steps):
                raise ValueError(
                    f"Each t_index_list value must be between 0 and num_inference_steps ({model.num_inference_steps}). Found {value} at index {i}."
                )

        for i in range(1, len(model.t_index_list)):
            curr, prev = model.t_index_list[i], model.t_index_list[i - 1]
            if curr < prev:
                raise ValueError(f"t_index_list must be in non-decreasing order. {curr} < {prev}")

        return model

    @model_validator(mode="after")
    @staticmethod
    def check_ip_adapter(model: "StreamDiffusionParams") -> "StreamDiffusionParams":
        supported = get_model_type(model.model_id) in IPADAPTER_SUPPORTED_TYPES
        enabled = model.ip_adapter and model.ip_adapter.enabled
        if not supported and enabled:
            raise ValueError(f"IPAdapter is not supported for {model.model_id}")
        return model

    @model_validator(mode="after")
    @staticmethod
    def check_controlnets(model: "StreamDiffusionParams") -> "StreamDiffusionParams":
        if not model.controlnets:
            return model

        cn_ids = set()
        for cn in model.controlnets:
            if cn.model_id in cn_ids:
                raise ValueError(f"Duplicate controlnet model_id: {cn.model_id}")
            cn_ids.add(cn.model_id)

        model_type = get_model_type(model.model_id)
        supported_cns = CONTROLNETS_BY_TYPE.get(model_type, [])

        invalid_cns = [cn for cn in cn_ids if cn not in supported_cns]
        if invalid_cns:
            raise ValueError(f"Invalid ControlNets for model {model.model_id}: {invalid_cns}")

        if model_type == "sdxl" and not _is_building_tensorrt_engines:
            enabled_cns = [
                cn for cn in model.controlnets
                if cn.enabled and cn.conditioning_scale > 0
            ]
            if len(enabled_cns) > 3:
                raise ValueError(
                    f"SDXL models support a maximum of 3 enabled ControlNets, found {len(enabled_cns)}."
                )

        return model

    @model_validator(mode="after")
    @staticmethod
    def check_cached_attention(model: "StreamDiffusionParams") -> "StreamDiffusionParams":
        cfg = model.cached_attention
        if not cfg or not cfg.enabled:
            return model

        if model.acceleration != "tensorrt":
            raise ValueError("Cached attention is only supported when acceleration='tensorrt'")

        if model.width != 512 or model.height != 512:
            raise ValueError("Cached attention currently supports only 512x512 resolution")

        return model


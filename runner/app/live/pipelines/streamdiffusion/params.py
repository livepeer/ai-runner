from typing import Dict, List, Literal, Optional, Any, Tuple, TypeVar, Generic

from pydantic import BaseModel, model_validator, Field

from ..interface import BaseParams

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
    """
    Generic preprocessor configuration model.

    Type parameter ProcessorTypeT should be a Literal type defining the available processor types.
    """
    class Config:
        extra = "forbid"

    type: ProcessorTypeT
    """Type of the preprocessor."""

    enabled: bool = True
    """Whether this preprocessor is active."""

    # Library has an "order" field, but we simply populate it with the index of the processor in the list
    # order: int

    params: ProcessorParams = Field(default_factory=dict)
    """Parameters for the preprocessor."""

class ProcessingConfig(BaseModel, Generic[ProcessorTypeT]):
    """
    Generic image and latent preprocessing configuration model.
    """
    class Config:
        extra = "forbid"

    enabled: bool = True
    """Whether this preprocessing is active."""

    processors: List[SingleProcessorConfig[ProcessorTypeT]] = Field(default_factory=list)
    """List of processors to apply."""

class ControlNetConfig(BaseModel):
    """
    ControlNet configuration model for guided image generation.

    **Dynamic updates limited to conditioning_scale changes only; cannot add
    new ControlNets or change model_id/preprocessor/params without reload.**
    """
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
    """ControlNet model identifier. Each model provides different types of conditioning:
    - openpose: Human pose estimation for figure control
    - hed: Soft edge/inner contour guidance; captures gentle gradients rather than crisp outlines
    - canny: Crisp silhouette/edge guidance; follows strong, high-contrast boundaries
    - depth: Depth guidance for 3D structure and spatial layout
    - color: Increases adherence/pass-through to the input's color palette (raise to keep colors)
    - tile: Detail refinement through tiling to enhance local texture and preserve structure on low-resolution inputs"""

    conditioning_scale: float = 1.0
    """Strength of the ControlNet's influence on generation. Higher values make the model follow the control signal more strictly. Typical range 0.0-1.0, where 0.0 disables the control and 1.0 applies full control."""

    conditioning_channels: int | None = Field(
        default=None, # model-specific default derived when constructing params
        description="Number of channels in the controlnet's conditioning input tensor. Defaults to 6 for TemporalNets and 3 for others.",
        ge=1,
        le=6
    )

    preprocessor: ImageProcessorName = "passthrough"
    """Preprocessor to apply to input frames before feeding to the ControlNet. Common options include 'pose_tensorrt', 'soft_edge', 'canny', 'depth_tensorrt', 'passthrough'. If None, no preprocessing is applied."""

    preprocessor_params: ProcessorParams = Field(default_factory=dict)
    """Additional parameters for the preprocessor. For example, canny edge detection uses 'low_threshold' and 'high_threshold' values."""

    enabled: bool = True
    """Whether this ControlNet is active. Disabled ControlNets are not loaded."""

    control_guidance_start: float = 0.0
    """Fraction of the denoising process (0.0-1.0) when ControlNet guidance begins. 0.0 means guidance starts from the beginning."""

    control_guidance_end: float = 1.0
    """Fraction of the denoising process (0.0-1.0) when ControlNet guidance ends. 1.0 means guidance continues until the end."""

class IPAdapterConfig(BaseModel):
    """
    IPAdapter configuration for style transfer.
    """
    class Config:
        extra = "forbid"

    type: Literal["regular", "faceid"] = "regular"
    """Type of IPAdapter to use. FaceID is used for face-specific style transfer."""

    ipadapter_model_path: Optional[Literal[
        "h94/IP-Adapter/models/ip-adapter_sd15.bin",
        "h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin",
        "h94/IP-Adapter-FaceID/ip-adapter-faceid_sd15.bin",
        "h94/IP-Adapter-FaceID/ip-adapter-faceid_sdxl.bin",
    ]] = None
    """[DEPRECATED] This field is no longer used. The IPAdapter model path is automatically determined based on the IP-Adapter type and diffusion model type."""

    image_encoder_path: Optional[Literal[
        "h94/IP-Adapter/models/image_encoder",
        "h94/IP-Adapter/sdxl_models/image_encoder",
    ]] = None
    """[DEPRECATED] This field is no longer used. The image encoder path is automatically determined based on the IP-Adapter type and diffusion model type."""

    insightface_model_name: Optional[Literal["buffalo_l"]] = "buffalo_l"
    """InsightFace model name for FaceID. Used only if type is 'faceid'."""

    scale: float = 1.0
    """IPAdapter strength (0.0 = disabled, 1.0 = normal, 2.0 = strong)"""

    weight_type: Optional[Literal[
        "linear", "ease in", "ease out", "ease in-out", "reverse in-out",
        "weak input", "weak output", "weak middle", "strong middle",
        "style transfer", "composition", "strong style transfer",
        "style and composition", "style transfer precise", "composition precise"
    ]] = "linear"
    """Weight distribution type for per-layer scaling"""

    enabled: bool = True
    """Whether this IPAdapter is active"""


class CachedAttentionConfig(BaseModel):
    """
    Cached attention (StreamV2V) configuration.
    """

    class Config:
        extra = "forbid"

    enabled: bool = True
    """Enable cached attention to reuse key/value tensors across frames."""

    max_frames: int = Field(
        default=1,
        ge=CACHED_ATTENTION_MIN_FRAMES,
        le=CACHED_ATTENTION_MAX_FRAMES,
        description="Number of historical K/V frames to retain. Limited by TensorRT engine exports.",
    )
    """Number of frames retained in the attention cache."""

    interval: int = Field(
        default=1,
        ge=1,
        le=1440, # 1 minute @ 24 FPS
        description="How often (in number of frames) to refresh the cache.",
    )
    """Cadence (number of frames) for refreshing cached key/value tensors."""



class StreamDiffusionParams(BaseParams):
    """
    StreamDiffusion pipeline parameters.

    **Dynamically updatable parameters** (no reload required):
    - prompt, guidance_scale, delta, num_inference_steps, t_index_list, seed,
      controlnets.conditioning_scale, cached_attention.max_frames, cached_attention.interval

    All other parameters require a full pipeline reload when changed.
    """
    class Config:
        extra = "forbid"

    # Model configuration
    model_id: Literal[
        "stabilityai/sd-turbo",
        "stabilityai/sdxl-turbo",
        "prompthero/openjourney-v4",
        "Lykon/dreamshaper-8",
    ] = "stabilityai/sd-turbo"
    """Base U-Net model to use for generation."""

    # Generation parameters
    prompt: str | List[Tuple[str, float]] = "flowers"
    """Text prompt describing the desired image. Can be a single string or weighted list of (prompt, weight) tuples."""

    prompt_interpolation_method: Literal["linear", "slerp"] = "slerp"
    """Method for interpolating between multiple prompts. Slerp provides smoother transitions than linear."""

    normalize_prompt_weights: bool = True
    """Whether to normalize prompt weights to sum to 1.0 for consistent generation."""

    negative_prompt: str = "blurry, low quality, flat, 2d"
    """Text describing what to avoid in the generated image."""

    guidance_scale: float = 1.0
    """Strength of prompt adherence. Higher values make the model follow the prompt more strictly."""

    delta: float = 0.7
    """Delta sets per-frame denoising progress: lower delta means steadier, less flicker but slower/softer; higher delta means faster, sharper but more flicker/artifacts (often reduce CFG)."""

    num_inference_steps: int = Field(
        default=50,
        ge=1,
        le=100,
        description='Builds the full denoising schedule (the "grid" of possible refinement steps). Changing it changes what each step number (t_index_list value) means. Keep it fixed for a session and only adjust if you\'re deliberately redefining the schedule; if you do, proportionally remap your t_index_list. Range: 1–100 with default being 50.'
    )

    t_index_list: List[int] = Field(default_factory=lambda: [12, 20, 32])
    """The ordered list of step indices from the num_inference_steps schedule to execute per frame. Each index is one model pass, so latency scales with the list length. Higher indices (e.g., 40–49 on a 50-step grid) mainly polish and preserve structure (lower flicker), while lower indices (<20) rewrite structure (more flicker, creative). Values must be non-decreasing, and each between 0 and num_inference_steps."""

    # LoRA settings
    lora_dict: Optional[Dict[str, float]] = None
    """Dictionary mapping LoRA model paths to their weights for fine-tuning the base model."""

    use_lcm_lora: bool = True
    """Whether to use Latent Consistency Model LoRA for faster inference."""

    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    """Identifier for the LCM LoRA model to use."""

    # Acceleration settings
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    """Acceleration method for inference. TensorRT provides the best performance but requires engine compilation."""

    # Processing settings
    use_safety_checker: bool = True
    """Whether to use the safety checker to prevent generating NSFW images."""

    safety_checker_threshold: float = 0.5
    """Threshold for the safety checker. Higher values allow more NSFW images to passthrough."""

    use_denoising_batch: bool = True
    """Whether to process multiple denoising steps in a single batch for efficiency."""

    do_add_noise: bool = True
    """Whether to add noise to input frames before processing. Enabling this slightly re-noises each frame to improve temporal stability, reduce ghosting/texture sticking, and prevent drift; disabling can yield sharper, lower-latency results but may increase flicker and artifact accumulation over time."""

    skip_diffusion: bool = False
    """Whether to skip diffusion and apply only preprocessing/postprocessing hooks. When True, bypasses VAE encoding, diffusion, and VAE decoding, but still applies image preprocessing and postprocessing hooks for consistent processing."""

    seed: int | List[Tuple[int, float]] = 789
    """Random seed for generation. Can be a single integer or weighted list of (seed, weight) tuples."""

    seed_interpolation_method: Literal["linear", "slerp"] = "linear"
    """Method for interpolating between multiple seeds. Slerp provides smoother transitions than linear."""

    normalize_seed_weights: bool = True
    """Whether to normalize seed weights to sum to 1.0 for consistent generation."""

    # Similar image filter settings
    enable_similar_image_filter: bool = False
    """Whether to skip frames that are too similar to the previous output to reduce flicker."""

    similar_image_filter_threshold: float = 0.98
    """Similarity threshold for the image filter. Higher values allow more variation between frames."""

    similar_image_filter_max_skip_frame: int = 10
    """Maximum number of consecutive frames that can be skipped by the similarity filter."""

    # ControlNet settings
    controlnets: List[ControlNetConfig] = Field(default_factory=list)
    """List of ControlNet configurations for guided generation. Each ControlNet provides different types of conditioning (pose, edges, depth, etc.)."""

    # IPAdapter settings
    ip_adapter: IPAdapterConfig = Field(default_factory=lambda: IPAdapterConfig(enabled=False))
    """IPAdapter configuration for style transfer."""

    ip_adapter_style_image_url: str = "https://storage.googleapis.com/lp-ai-assets/ipadapter_style_imgs/textures/vortex.jpeg"
    """URL to fetch the style image for IPAdapter."""

    # Processors

    image_preprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    """List of image preprocessor configurations for image processing."""

    image_postprocessing: Optional[ProcessingConfig[ImageProcessorName]] = None
    """List of image postprocessor configurations for image processing."""

    latent_preprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None
    """List of latent preprocessor configurations for latent processing."""

    latent_postprocessing: Optional[ProcessingConfig[LatentProcessorsName]] = None
    """List of latent postprocessor configurations for latent processing."""

    cached_attention: CachedAttentionConfig = Field(default_factory=lambda: CachedAttentionConfig(enabled=True))
    """Cached attention configuration."""

    def get_output_resolution(self) -> tuple[int, int]:
        """
        Get the output resolution as a (width, height) tuple, accounting for upscale processors
        in image_postprocessing.
        """
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

        # SDXL models can only have up to 3 enabled controlnets due to VRAM limitations on 4090s
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

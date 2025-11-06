from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .streamdiffusion_processors_params import (
    BaseProcessorParams,
    ControlNetPreprocessorLiteral,
    ImageProcessorLiteral,
    LatentProcessorLiteral,
    preprocessors,
)

from .interface import BaseParams

ModelType = Literal["sd15", "sd21", "sdxl"]

IPADAPTER_SUPPORTED_TYPES: List[ModelType] = ["sd15", "sdxl"]

CONTROLNETS_BY_TYPE: Dict[ModelType, List[str]] = {
    "sd21": [
        "thibaud/controlnet-sd21-openpose-diffusers",
        "thibaud/controlnet-sd21-hed-diffusers",
        "thibaud/controlnet-sd21-canny-diffusers",
        "thibaud/controlnet-sd21-depth-diffusers",
        "thibaud/controlnet-sd21-color-diffusers",
        "thibaud/controlnet-sd21-ade20k-diffusers",
        "thibaud/controlnet-sd21-normalbae-diffusers",
        "daydreamlive/TemporalNet2-stable-diffusion-2-1",
    ],
    "sd15": [
        "lllyasviel/control_v11f1p_sd15_depth",
        "lllyasviel/control_v11f1e_sd15_tile",
        "lllyasviel/control_v11p_sd15_canny",
        "CiaraRowles/TemporalNet2",
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


ProcessorRegistry = Dict[str, Type[BaseProcessorParams]]


class ProcessorInvocation(BaseModel):
    """Represents a single processor invocation with enable/order metadata."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    order: Optional[int] = None
    params: BaseProcessorParams

    def to_wrapper_entry(self, name: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"type": name, "enabled": self.enabled}
        if self.order is not None:
            entry["order"] = self.order
        params_payload = self.params.model_dump(exclude_none=True)
        if params_payload:
            entry["params"] = params_payload
        return entry


def _coerce_processor_params(
    name: str,
    value: Any,
    registry: ProcessorRegistry,
) -> BaseProcessorParams:
    config_cls = registry.get(name)
    if config_cls is None:
        raise ValueError(f"Unsupported processor '{name}'")

    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, BaseProcessorParams):
        return config_cls(**value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return config_cls(**value)

    raise TypeError(
        f"Processor '{name}' parameters must be a dict or {config_cls.__name__}, "
        f"got {type(value).__name__}"
    )


def _normalize_processor_map(
    raw_value: Any,
    registry: ProcessorRegistry,
    domain: str,
) -> Dict[str, ProcessorInvocation]:
    if raw_value is None:
        return {}

    if isinstance(raw_value, ProcessorHookConfig):
        # Already normalized - just reuse the backing dictionary.
        return dict(raw_value.processors)

    items: List[Tuple[str, Any]]

    if isinstance(raw_value, dict):
        items = list(raw_value.items())
    elif isinstance(raw_value, list):
        items = []
        for entry in raw_value:
            if not isinstance(entry, dict):
                raise TypeError(
                    f"{domain.capitalize()} processors must be supplied as dictionaries; "
                    f"received {type(entry).__name__}"
                )
            if "type" not in entry:
                raise ValueError(f"{domain.capitalize()} processor entries must include a 'type'")
            entry_copy = dict(entry)
            proc_type = entry_copy.pop("type")
            items.append((proc_type, entry_copy))
    else:
        raise TypeError(
            f"{domain.capitalize()} processors must be defined via a mapping or list, "
            f"received {type(raw_value).__name__}"
        )

    normalized: Dict[str, ProcessorInvocation] = {}
    for name, payload in items:
        if name in normalized:
            raise ValueError(f"Duplicate {domain} processor '{name}'")

        if isinstance(payload, ProcessorInvocation):
            normalized[name] = payload
            continue

        if payload is None:
            payload_dict: Dict[str, Any] = {}
        elif isinstance(payload, dict):
            payload_dict = dict(payload)
        else:
            # Treat bare params as shorthand for {"params": <value>}
            payload_dict = {"params": payload}

        enabled = payload_dict.pop("enabled", True)
        order = payload_dict.pop("order", None)
        params_data = payload_dict.pop("params", None)

        if payload_dict:
            unexpected = ", ".join(sorted(payload_dict.keys()))
            raise ValueError(f"Unexpected fields for {domain} '{name}': {unexpected}")

        params = _coerce_processor_params(name, params_data, registry)
        normalized[name] = ProcessorInvocation(enabled=enabled, order=order, params=params)

    return normalized


class ProcessorHookConfig(BaseModel):
    """Typed map of processor name -> invocation metadata."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    processors: Dict[str, ProcessorInvocation] = Field(default_factory=dict)

    registry: ClassVar[ProcessorRegistry] = {}
    domain: ClassVar[str] = "processor"

    @field_validator("processors", mode="before")
    @classmethod
    def _validate_processors(cls, value: Any) -> Dict[str, ProcessorInvocation]:
        return _normalize_processor_map(value, cls.registry, cls.domain)

    def to_wrapper_payload(self) -> Dict[str, Any]:
        ordered = list(enumerate(self.processors.items()))
        ordered.sort(
            key=lambda item: (
                item[1][1].order if item[1][1].order is not None else item[0],
                item[1][0],
            )
        )

        payload = {
            "enabled": self.enabled,
            "processors": [
                invocation.to_wrapper_entry(name) for _, (name, invocation) in ordered
            ],
        }
        return payload

    def iter_requires_pipeline(self) -> List[str]:
        return [
            name
            for name in self.processors
            if self.registry[name].requires_pipeline_ref
        ]


class ImageProcessorHookConfig(ProcessorHookConfig):
    processors: Dict[ImageProcessorLiteral, ProcessorInvocation] = Field(default_factory=dict)
    registry: ClassVar[ProcessorRegistry] = preprocessors.IMAGE_REGISTRY
    domain: ClassVar[str] = "image processor"


class LatentProcessorHookConfig(ProcessorHookConfig):
    processors: Dict[LatentProcessorLiteral, ProcessorInvocation] = Field(default_factory=dict)
    registry: ClassVar[ProcessorRegistry] = preprocessors.LATENT_REGISTRY
    domain: ClassVar[str] = "latent processor"

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
        "thibaud/controlnet-sd21-ade20k-diffusers",
        "thibaud/controlnet-sd21-normalbae-diffusers",
        "daydreamlive/TemporalNet2-stable-diffusion-2-1",
        "lllyasviel/control_v11f1p_sd15_depth",
        "lllyasviel/control_v11f1e_sd15_tile",
        "lllyasviel/control_v11p_sd15_canny",
        "CiaraRowles/TemporalNet2",
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

    preprocessor: ControlNetPreprocessorLiteral = "passthrough"
    """Preprocessor to apply to input frames before feeding to the ControlNet. Common options include 'pose_tensorrt', 'soft_edge', 'canny', 'depth_tensorrt', 'passthrough'. If None, no preprocessing is applied."""
    # IPAdapter embedding processors are intentionally excluded here—they are handled through the IPAdapter subsystem.

    preprocessor_params: BaseProcessorParams | Dict[str, Any] | None = None
    """Typed parameters for the configured preprocessor."""

    enabled: bool = True
    """Whether this ControlNet is active. Disabled ControlNets are not loaded."""

    control_guidance_start: float = 0.0
    """Fraction of the denoising process (0.0-1.0) when ControlNet guidance begins. 0.0 means guidance starts from the beginning."""

    control_guidance_end: float = 1.0
    """Fraction of the denoising process (0.0-1.0) when ControlNet guidance ends. 1.0 means guidance continues until the end."""

    @model_validator(mode="after")
    def _coerce_preprocessor_params(self) -> "ControlNetConfig":
        params = self.preprocessor_params
        registry = preprocessors.CONTROLNET_REGISTRY
        params = _coerce_processor_params(self.preprocessor, params, registry)
        self.preprocessor_params = params
        return self

_DEFAULT_CONTROLNETS = [
    ControlNetConfig(
        model_id="thibaud/controlnet-sd21-openpose-diffusers",
        conditioning_scale=0.711,
        preprocessor="pose_tensorrt",
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
    ControlNetConfig(
        model_id="thibaud/controlnet-sd21-hed-diffusers",
        conditioning_scale=0.2,
        preprocessor="soft_edge",
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
    ControlNetConfig(
        model_id="thibaud/controlnet-sd21-canny-diffusers",
        conditioning_scale=0.2,
        preprocessor="canny",
        preprocessor_params=preprocessors.Canny(
            low_threshold=100,
            high_threshold=200,
        ),
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
    ControlNetConfig(
        model_id="thibaud/controlnet-sd21-depth-diffusers",
        conditioning_scale=0.5,
        preprocessor="depth_tensorrt",
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
    ControlNetConfig(
        model_id="thibaud/controlnet-sd21-color-diffusers",
        conditioning_scale=0.2,
        preprocessor="passthrough",
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
    ControlNetConfig(
        model_id="daydreamlive/TemporalNet2-stable-diffusion-2-1",
        conditioning_scale=0.0,
        preprocessor="temporal_net_tensorrt",
        preprocessor_params=preprocessors.TemporalNetTensorRT(flow_strength=0.4),
        enabled=True,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ),
]
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


class StreamDiffusionParams(BaseParams):
    """
    StreamDiffusion pipeline parameters.

    **Dynamically updatable parameters** (no reload required):
    - prompt, guidance_scale, delta, num_inference_steps, t_index_list, seed,
      controlnets.conditioning_scale

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

    num_inference_steps: int = 50
    """Builds the full denoising schedule (the "grid" of possible refinement steps). Changing it changes what each step number (t_index_list value) means. Keep it fixed for a session and only adjust if you're deliberately redefining the schedule; if you do, proportionally remap your t_index_list. Typical range 10–200 with default being 50."""

    t_index_list: List[int] = [12, 20, 32]
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
    controlnets: Optional[List[ControlNetConfig]] = _DEFAULT_CONTROLNETS
    """List of ControlNet configurations for guided generation. Each ControlNet provides different types of conditioning (pose, edges, depth, etc.)."""

    # Pipeline hook settings
    image_preprocessing: Optional[ImageProcessorHookConfig] = None
    """Typed image-domain preprocessing chain applied before the main pipeline."""

    image_postprocessing: Optional[ImageProcessorHookConfig] = None
    """Typed image-domain postprocessing chain applied after pipeline outputs."""

    latent_preprocessing: Optional[LatentProcessorHookConfig] = None
    """Typed latent-domain preprocessing chain. Feedback processors may require additional normalization depending on hook placement."""

    # IPAdapter settings
    ip_adapter: Optional[IPAdapterConfig] = IPAdapterConfig(enabled=False)
    """IPAdapter configuration for style transfer."""

    ip_adapter_style_image_url: str = "https://storage.googleapis.com/lp-ai-assets/ipadapter_style_imgs/textures/vortex.jpeg"
    """URL to fetch the style image for IPAdapter."""

    def build_hook_payloads(self) -> Dict[str, Any]:
        """Translate typed processor hooks into the wrapper's expected payload shape."""
        payload: Dict[str, Any] = {}
        if self.image_preprocessing:
            payload["image_preprocessing_config"] = self.image_preprocessing.to_wrapper_payload()
        if self.image_postprocessing:
            payload["image_postprocessing_config"] = self.image_postprocessing.to_wrapper_payload()
        if self.latent_preprocessing:
            payload["latent_preprocessing_config"] = self.latent_preprocessing.to_wrapper_payload()
        return payload

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

        return model

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set

import torch
from huggingface_hub import hf_hub_download

from .params import (
    MODEL_ID_TO_TYPE,
    CONTROLNETS_BY_TYPE,
    IPADAPTER_SUPPORTED_TYPES,
    StreamDiffusionParams,
    ControlNetConfig,
    IPAdapterConfig,
    ProcessingConfig,
    SingleProcessorConfig,
    ModelType,
    CachedAttentionConfig,
    CACHED_ATTENTION_MIN_FRAMES,
    CACHED_ATTENTION_MAX_FRAMES,
)
from . import params
from .pipeline import load_streamdiffusion_sync, ENGINES_DIR, LOCAL_MODELS_DIR

MIN_TIMESTEPS = 1
MAX_TIMESTEPS = 4

# Optimal number of timesteps for t_index_list per model type
OPTIMAL_TIMESTEPS_BY_TYPE: Dict[ModelType, int] = {
    "sd15": 3,
    "sd21": 3,
    "sdxl": 2,
}

# Which model types to build for each SUBVARIANT.
# This allows splitting engine builds across different container images.
# See .github/workflows/ai-runner-docker-live-streamdiffusion.yaml for the list of subvariants.
SUBVARIANT_MODEL_TYPES: Dict[str, List[ModelType]] = {
    "": ["sd15", "sd21", "sdxl"],    # Empty SUBVARIANT: builds ALL models (for public operators)
    "sdturbo": ["sd15", "sd21"],     # SD Turbo variant (sd21) + SD 1.5
    "sd15": ["sd15", "sd21"],        # SD 1.5 variant
    "sd15-v2v": ["sd15", "sd21"],    # SD 1.5 with streamv2v variant
    "sdxl": ["sdxl"],                # SDXL variant
    "sdxl-faceid": ["sdxl"],         # SDXL with FaceID variant
}

def _get_allowed_model_types() -> Set[ModelType]:
    """Get the set of model types to build based on SUBVARIANT env var."""
    subvariant = os.environ.get("SUBVARIANT", "")

    if subvariant in SUBVARIANT_MODEL_TYPES:
        allowed = set(SUBVARIANT_MODEL_TYPES[subvariant])
        logging.info("SUBVARIANT=%r: building model types %s", subvariant, sorted(allowed))
        return allowed

    # Unknown SUBVARIANT = build everything with a warning
    all_types: Set[ModelType] = set(MODEL_ID_TO_TYPE.values())
    logging.warning("SUBVARIANT=%r is unknown, building ALL model types %s", subvariant, sorted(all_types))
    return all_types

MIN_RESOLUTION = 384
MAX_RESOLUTION = 1024

BASE_GIT_REPOS_DIR = Path("/workspace")

# Models directory is fixed in Docker image via HUGGINGFACE_HUB_CACHE env var
MODELS_DIR = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/models"))

@dataclass(frozen=True)
class GitRepo:
    url: str
    commit: str


@dataclass(frozen=True)
class HfAsset:
    repo_id: str
    filename: str


DEPTH_EXPORT_REPO = GitRepo(
    url="https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt.git",
    commit="1f4c161949b3616516745781fb91444e6443cc25",
)
POSE_EXPORT_REPO = GitRepo(
    url="https://github.com/yuvraj108c/ComfyUI-YoloNasPose-Tensorrt.git",
    commit="873de560bb05bf3331e4121f393b83ecc04c324a",
)

DEPTH_ONNX_MODEL = HfAsset(
    repo_id="yuvraj108c/Depth-Anything-2-Onnx",
    filename="depth_anything_v2_vits.onnx",
)
POSE_ASSETS: Sequence[HfAsset] = (
    HfAsset(
        repo_id="yuvraj108c/yolo-nas-pose-onnx",
        filename="yolo_nas_pose_l_0.5.onnx",
    ),
    # might want to add other confidence thresholds later
)


@dataclass(frozen=True)
class BuildJob:
    params: StreamDiffusionParams


def prepare_streamdiffusion_models() -> None:
    params._is_building_tensorrt_engines = True

    if not ENGINES_DIR.exists():
        raise ValueError(f"Engines dir ({ENGINES_DIR}) does not exist")
    if not LOCAL_MODELS_DIR.exists():
        raise ValueError(f"Local models dir ({LOCAL_MODELS_DIR}) does not exist")

    logging.info("Preparing StreamDiffusion assets in %s", MODELS_DIR)
    _compile_dependencies()
    jobs = list(_build_matrix())
    logging.info("Compilation plan has %d build(s)", len(jobs))
    for idx, job in enumerate(jobs, start=1):
        logging.info(
            "[%s/%s] Compiling model=%s ipadapter=%s %sx%s",
            idx,
            len(jobs),
            job.params.model_id,
            job.params.ip_adapter.type if job.params.ip_adapter and job.params.ip_adapter.enabled else "disabled",
            job.params.width,
            job.params.height,
        )
        _compile_build(job)
    logging.info("StreamDiffusion model preparation complete.")


def _compile_dependencies() -> None:
    _build_depth_anything()
    _build_pose_engines()
    _build_raft_engine()


def _build_depth_anything() -> None:
    engine_path = ENGINES_DIR / "depth-anything" / "depth_anything_v2_vits.engine"
    if engine_path.exists():
        logging.info("Depth-Anything engine already present: %s", engine_path)
        return
    engine_path = engine_path.resolve()

    logging.info("Building Depth-Anything TensorRT engine...")
    repo_dir = _ensure_repo(DEPTH_EXPORT_REPO)
    onnx_path = _download_asset(DEPTH_ONNX_MODEL)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "export_trt.py",
            "--trt-path",
            str(engine_path),
            "--onnx-path",
            str(onnx_path),
        ],
        cwd=repo_dir,
        check=True,
    )
    logging.info("Depth-Anything engine written to %s", engine_path)


def _build_pose_engines() -> None:
    repo_dir = _ensure_repo(POSE_EXPORT_REPO)
    requirements = repo_dir / "requirements.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=repo_dir,
        check=True,
    )

    for asset in POSE_ASSETS:
        onnx_path = _download_asset(asset)
        engine_name = Path(asset.filename).with_suffix(".engine").name
        engine_path = ENGINES_DIR / "pose" / engine_name
        if engine_path.exists():
            logging.info("Pose engine already present: %s", engine_path)
            continue

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Building pose engine for %s", onnx_path.name)
        link_target = repo_dir / "yolo_nas_pose_l.onnx"
        if link_target.exists() or link_target.is_symlink():
            link_target.unlink()
        link_target.symlink_to(onnx_path)

        subprocess.run(
            [sys.executable, "export_trt.py"],
            cwd=repo_dir,
            check=True,
        )
        produced = repo_dir / "yolo_nas_pose_l.engine"
        if not produced.exists():
            raise RuntimeError("Pose exporter did not produce expected engine file")
        shutil.move(str(produced), str(engine_path))
        logging.info("Pose engine written to %s", engine_path)


def _build_raft_engine() -> None:
    engine_path = ENGINES_DIR / "temporal_net" / "raft_small_min_384x384_max_1024x1024.engine"
    if engine_path.exists():
        logging.info("RAFT engine already present: %s", engine_path)
        return

    logging.info("Compiling RAFT TensorRT engine...")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamdiffusion.tools.compile_raft_tensorrt",
            "--min_resolution",
            f"{MIN_RESOLUTION}x{MIN_RESOLUTION}",
            "--max_resolution",
            f"{MAX_RESOLUTION}x{MAX_RESOLUTION}",
            "--output_dir",
            str(engine_path.parent),
        ],
        check=True,
    )
    logging.info("RAFT engine written to %s", engine_path)


def _ensure_repo(repo: GitRepo) -> Path:
    repo_name = Path(repo.url).name.replace(".git", "")
    repo_dir = BASE_GIT_REPOS_DIR / repo_name
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", repo.url, str(repo_dir)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", repo.commit],
        check=True,
    )
    return repo_dir


def _build_matrix() -> Iterator[BuildJob]:
    allowed_types = _get_allowed_model_types()

    for model_id, model_type in MODEL_ID_TO_TYPE.items():
        if model_type not in allowed_types:
            logging.info("Skipping model %s (type=%s): not in allowed types for this SUBVARIANT", model_id, model_type)
            continue

        ipa_types: Sequence[Optional[str]]
        ipa_types = [None]
        if model_type in IPADAPTER_SUPPORTED_TYPES:
            ipa_types = ["regular", "faceid"]

        for ipa_type in ipa_types:
            for use_cached_attn in (False, True):
                params = _create_params(model_id, model_type, ipa_type, use_cached_attn)
                yield BuildJob(params=params)


def _compile_build(job: BuildJob) -> None:
    controlnet_ids = [cn.model_id for cn in job.params.controlnets] if job.params.controlnets else []
    print(
        f"→ Building TensorRT engines | model={job.params.model_id} "
        f"ipadapter={job.params.ip_adapter.type if job.params.ip_adapter and job.params.ip_adapter.enabled else 'disabled'} "
        f"size={job.params.width}x{job.params.height} "
        f"cached_attention={'on' if job.params.cached_attention.enabled else 'off'} "
        f"timesteps={job.params.t_index_list} batch_min={MIN_TIMESTEPS} batch_max={MAX_TIMESTEPS} "
        f"controlnets={controlnet_ids}"
    )
    try:
        pipe = load_streamdiffusion_sync(
            params=job.params,
            min_batch_size=MIN_TIMESTEPS,
            max_batch_size=MAX_TIMESTEPS,
            engine_dir=ENGINES_DIR,
            build_engines=True,
        )
        # Explicitly drop the wrapper to keep GPU memory low between builds.
        del pipe
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _create_params(
    model_id: str,
    model_type: ModelType,
    ipa_type: Optional[str],
    use_cached_attn: bool,
) -> StreamDiffusionParams:
    controlnets = []
    controlnet_ids = CONTROLNETS_BY_TYPE.get(model_type)
    if controlnet_ids:
        for cn_model_id in controlnet_ids:
            preprocessor = "passthrough" if "TemporalNet" not in cn_model_id else "temporal_net_tensorrt"
            config = ControlNetConfig(
                model_id=cn_model_id,  # type: ignore
                conditioning_scale=0.5,
                preprocessor=preprocessor,
                preprocessor_params={},
                enabled=True,
                control_guidance_start=0.0,
                control_guidance_end=1.0,
            )
            controlnets.append(config)

    # Create t_index_list based on number of timesteps. Only the size matters... 😏
    step_count = OPTIMAL_TIMESTEPS_BY_TYPE.get(model_type, 3)
    t_index_list = list(range(1, 50, 50 // step_count))[:step_count]

    return StreamDiffusionParams(
        model_id=model_id,  # type: ignore
        width=512,
        height=512,
        acceleration="tensorrt",
        t_index_list=t_index_list,
        controlnets=controlnets,
        ip_adapter=IPAdapterConfig(
            enabled=ipa_type is not None,
            type="faceid" if ipa_type == "faceid" else "regular",
        ),
        image_postprocessing=ProcessingConfig(
            processors=[SingleProcessorConfig(type="realesrgan_trt")]
        ),
        cached_attention=CachedAttentionConfig(enabled=use_cached_attn),
    )


def _download_asset(asset: HfAsset) -> Path:
    return Path(
        hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.filename,
            cache_dir=str(MODELS_DIR),
        )
    )


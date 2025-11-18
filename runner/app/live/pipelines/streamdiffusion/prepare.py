from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

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
)
from . import params
from .pipeline import load_streamdiffusion_sync, ENGINES_DIR, LOCAL_MODELS_DIR

MIN_TIMESTEPS = 1
MAX_TIMESTEPS = 4

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

    @property
    def engine_name(self) -> str:
        return Path(self.filename).with_suffix(".engine").name


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
)


@dataclass(frozen=True)
class BuildJob:
    model_id: str
    model_type: str
    ipadapter_type: Optional[str]
    width: int
    height: int


def prepare_streamdiffusion_models() -> None:
    params._is_building_tensorrt_engines = True

    if not Path(ENGINES_DIR).exists():
        raise ValueError(f"Engines dir ({ENGINES_DIR}) does not exist")
    if not Path(LOCAL_MODELS_DIR).exists():
        raise ValueError(f"Local models dir ({LOCAL_MODELS_DIR}) does not exist")

    logging.info("Preparing StreamDiffusion assets in %s", MODELS_DIR)
    _compile_dependencies()
    jobs = list(_build_matrix())
    logging.info("Compilation plan has %d build(s)", len(jobs))
    for idx, job in enumerate(jobs, start=1):
        logging.info(
            "[%s/%s] Compiling model=%s type=%s ipadapter=%s %sx%s",
            idx,
            len(jobs),
            job.model_id,
            job.model_type,
            job.ipadapter_type or "disabled",
            job.width,
            job.height,
        )
        _compile_build(job, Path(ENGINES_DIR))
    logging.info("StreamDiffusion model preparation complete.")


def _compile_dependencies() -> None:
    _build_depth_anything()
    _build_pose_engines()
    _build_raft_engine()


def _build_depth_anything() -> None:
    engine_path = Path(ENGINES_DIR) / "depth-anything" / "depth_anything_v2_vits.engine"
    if engine_path.exists():
        logging.info("Depth-Anything engine already present: %s", engine_path)
        return

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
    marker = repo_dir / ".requirements_installed"
    if requirements.exists() and not marker.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
            cwd=repo_dir,
            check=True,
        )
        marker.touch()

    engines_dir = Path(ENGINES_DIR)
    for asset in POSE_ASSETS:
        onnx_path = _download_asset(asset)
        engine_path = engines_dir / "pose" / asset.engine_name
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
    engines_dir = Path(ENGINES_DIR)
    engine_path = engines_dir / "temporal_net" / "raft_small_min_384x384_max_1024x1024.engine"
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
    control_keys = set(CONTROLNETS_BY_TYPE.keys())
    for model_id, model_type in MODEL_ID_TO_TYPE.items():
        ip_types: Sequence[Optional[str]]
        if model_type in IPADAPTER_SUPPORTED_TYPES:
            ip_types = ("regular", "faceid")
        else:
            ip_types = (None,)

        for ip_type in ip_types:
            if ip_type == "faceid" and model_type not in IPADAPTER_SUPPORTED_TYPES:
                continue
            if model_type not in control_keys:
                logging.warning("Unknown controlnet set for model type %s", model_type)
            base = _base_params(model_type, ip_type)
            yield BuildJob(
                model_id=model_id,
                model_type=model_type,
                ipadapter_type=ip_type,
                width=base.width,
                height=base.height,
            )


def _compile_build(job: BuildJob, engines_dir: Path) -> None:
    params = _params_for_job(job)
    controlnet_ids = [cn.model_id for cn in params.controlnets] if params.controlnets else []
    print(
        f"→ Building TensorRT engines | model={job.model_id} type={job.model_type} "
        f"ipadapter={job.ipadapter_type or 'disabled'} size={params.width}x{params.height} "
        f"timesteps={params.t_index_list} batch_min={MIN_TIMESTEPS} batch_max={MAX_TIMESTEPS} "
        f"controlnets={controlnet_ids}"
    )
    try:
        pipe = load_streamdiffusion_sync(
            params=params,
            min_batch_size=MIN_TIMESTEPS,
            max_batch_size=MAX_TIMESTEPS,
            engine_dir=str(engines_dir),
            build_engines=True,
        )
        # Explicitly drop the wrapper to keep GPU memory low between builds.
        del pipe
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _params_for_job(job: BuildJob) -> StreamDiffusionParams:
    base = _base_params(job.model_type, job.ipadapter_type)
    ip_adapter = _ipadapter_for_job(job, base)

    return base.model_copy(
        update={
            "model_id": job.model_id,
            "controlnets": _controlnets_for_type(job.model_type),
            "ip_adapter": ip_adapter,
            "image_postprocessing": ProcessingConfig(
                processors=[SingleProcessorConfig(type="realesrgan_trt")]
            ),
        },
        deep=True,
    )


def _base_params(model_type: str, ip_type: Optional[str]) -> StreamDiffusionParams:
    if model_type == "sd21":
        return StreamDiffusionParams()

    template_file = {
        "sd15": "sd15_default_params.json",
        "sdxl": "sdxl_faceid_default_params.json" if ip_type == "faceid" else "sdxl_default_params.json",
    }.get(model_type)

    if not template_file:
        return StreamDiffusionParams()

    template_path = Path(__file__).with_name(template_file)
    with template_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return StreamDiffusionParams(**data)


def _ipadapter_for_job(
    job: BuildJob, base: StreamDiffusionParams
) -> Optional[IPAdapterConfig]:
    if not job.ipadapter_type:
        return None

    base_cfg = base.ip_adapter or IPAdapterConfig()
    return base_cfg.model_copy(
        update={
            "type": job.ipadapter_type,
            "enabled": True,
        },
        deep=True,
    )


def _controlnets_for_type(model_type: str) -> Optional[List[ControlNetConfig]]:
    ids = CONTROLNETS_BY_TYPE.get(model_type)
    if not ids:
        return None

    templates = _controlnet_templates()
    configs: List[ControlNetConfig] = []
    for cn_id in ids:
        template = templates.get(cn_id)
        if not template:
            raise ValueError(f"No ControlNet template registered for {cn_id}")
        configs.append(template.model_copy(deep=True))
    return configs


_CONTROLNET_TEMPLATE_CACHE: Optional[Dict[str, ControlNetConfig]] = None


def _controlnet_templates() -> Dict[str, ControlNetConfig]:
    global _CONTROLNET_TEMPLATE_CACHE
    if _CONTROLNET_TEMPLATE_CACHE is not None:
        return _CONTROLNET_TEMPLATE_CACHE

    templates: Dict[str, ControlNetConfig] = {}

    def _ingest_params(params: StreamDiffusionParams):
        if not params.controlnets:
            return
        for cfg in params.controlnets:
            templates.setdefault(cfg.model_id, cfg)

    _ingest_params(StreamDiffusionParams())

    template_files = [
        "sd15_default_params.json",
        "sdxl_default_params.json",
        "sdxl_faceid_default_params.json",
    ]
    for filename in template_files:
        path = Path(__file__).with_name(filename)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        _ingest_params(StreamDiffusionParams(**data))

    _CONTROLNET_TEMPLATE_CACHE = templates
    return templates


def _download_asset(asset: HfAsset) -> Path:
    return Path(
        hf_hub_download(
            repo_id=asset.repo_id,
            filename=asset.filename,
            cache_dir=str(MODELS_DIR),
        )
    )


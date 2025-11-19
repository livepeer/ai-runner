#!/bin/bash

set -e
[ -v DEBUG ] && set -x

# ComfyUI image configuration
PULL_IMAGES=${PULL_IMAGES:-true}
AI_RUNNER_COMFYUI_IMAGE=${AI_RUNNER_COMFYUI_IMAGE:-livepeer/ai-runner:live-app-comfyui}
AI_RUNNER_STREAMDIFFUSION_IMAGE=${AI_RUNNER_STREAMDIFFUSION_IMAGE:-livepeer/ai-runner:live-app-streamdiffusion}
AI_RUNNER_SCOPE_IMAGE=${AI_RUNNER_SCOPE_IMAGE:-livepeer/ai-runner:live-app-scope}
CONDA_PYTHON="/workspace/miniconda3/envs/comfystream/bin/python"
PIPELINE=${PIPELINE:-all}

# Select a single NVIDIA GPU interactively and export NVIDIA_VISIBLE_DEVICES.
# Usage:
#   select_gpu && your_command_needing_cuda
#   # or later in the shell: echo "$NVIDIA_VISIBLE_DEVICES"
select_gpu() {
  # Ensure nvidia-smi is available
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found. Install NVIDIA drivers / CUDA toolkit." >&2
    return 2
  fi

  # Fetch GPU info (index, name, total/free memory, utilization)
  local IFS=$'\n'
  local rows=()
  mapfile -t rows < <(nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null)

  if ((${#rows[@]} == 0)); then
    echo "No NVIDIA GPUs detected." >&2
    return 3
  fi

  # Pretty-print a table
  echo "Available NVIDIA GPUs:"
  printf "  %-4s %-38s %10s %10s %8s\n" "IDX" "NAME" "VRAM(MB)" "FREE(MB)" "UTIL(%%)"
  local idx name tot free util
  for line in "${rows[@]}"; do
    IFS=',' read -r idx name tot free util <<<"$line"
    # trim spaces
    idx=${idx//[[:space:]]/}
    name=${name## }
    tot=${tot//[[:space:]]/}
    free=${free//[[:space:]]/}
    util=${util//[[:space:]]/}
    printf "  %-4s %-38s %10s %10s %8s\n" "$idx" "$name" "$tot" "$free" "$util"
  done

  # Prompt until a valid index is chosen
  local choice found=0 sel_name
  while :; do
    read -r -p "Select GPU index [0-$((${#rows[@]} - 1))] (or 'q' to cancel): " choice
    case "$choice" in
    q | Q)
      echo "Canceled."
      return 1
      ;;
    *[!0-9]* | "") echo "Please enter a valid numeric index." ;;
    *)
      found=0
      for line in "${rows[@]}"; do
        IFS=',' read -r idx name _ <<<"$line"
        idx=${idx//[[:space:]]/}
        if [[ "$idx" == "$choice" ]]; then
          sel_name=${name## }
          found=1
          break
        fi
      done
      if ((found)); then
        export NVIDIA_VISIBLE_DEVICES="$choice"
        echo "Using GPU $choice ($sel_name). Exported NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES"
        return 0
      else
        echo "GPU index '$choice' not found."
      fi
      ;;
    esac
  done
}

# Check HF_TOKEN and Hugging Face CLI login status, throw warning if not authenticated.
check_hf_auth() {
  if [ -z "$HF_TOKEN" ] && [ "$(hf auth whoami)" = "Not logged in" ]; then
    printf "WARN: Not logged in and HF_TOKEN not set. Log in with 'hf auth login' or set HF_TOKEN to download token-gated models.\n"
    exit 1
  fi
}

# Displays help message.
function display_help() {
  echo "Description: This script is used to download models available on the Livepeer AI Subnet."
  echo "Usage: $0 [--beta]"
  echo "Options:"
  echo "  --beta  Download beta models."
  echo "  --restricted  Download models with a restrictive license."
  echo "  --live  Download models only for the livestreaming pipelines."
  echo "  --tensorrt  Download livestreaming models and build tensorrt models."
  echo "  --batch  Download all models for batch processing."
  echo "  --help   Display this help message."
  echo ""
  echo "Environment Variables:"
  echo "  PULL_IMAGES  Whether to pull Docker images (default: true)"
  echo "  AI_RUNNER_COMFYUI_IMAGE  ComfyUI Docker image (default: livepeer/ai-runner:live-app-comfyui)"
  echo "  AI_RUNNER_STREAMDIFFUSION_IMAGE  StreamDiffusion Docker image (default: livepeer/ai-runner:live-app-streamdiffusion)"
  echo "  PIPELINE  When using --live or --tensorrt, specify which pipeline to use: 'streamdiffusion', 'comfyui', 'scope', or 'all' (default)"
  echo "  HF_TOKEN  HuggingFace token for downloading token-gated models"
  echo "  DEBUG  Enable debug mode with set -x"
}

# Download recommended models during beta phase.
function download_beta_models() {
  printf "\nDownloading recommended beta phase models...\n"

  printf "\nDownloading unrestricted models...\n"

  # Download text-to-image and image-to-image models.
  hf download SG161222/RealVisXL_V4.0_Lightning --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download ByteDance/SDXL-Lightning --include "*unet.safetensors" --cache-dir models
  hf download timbrooks/instruct-pix2pix --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --cache-dir models

  # Download upscale models
  hf download stabilityai/stable-diffusion-x4-upscaler --include "*.fp16.safetensors" --cache-dir models

  # Download audio-to-text models.
  hf download openai/whisper-large-v3 --include "*.safetensors" --include "*.json" --cache-dir models
  hf download distil-whisper/distil-large-v3 --include "*.safetensors" --include "*.json" --cache-dir models
  hf download openai/whisper-medium --include "*.safetensors" --include "*.json" --cache-dir models

  # Download custom pipeline models.
  hf download facebook/sam2-hiera-large --include "*.pt" --include "*.yaml" --cache-dir models
  hf download parler-tts/parler-tts-large-v1 --include "*.safetensors" --include "*.json" --include "*.model" --cache-dir models

  printf "\nDownloading token-gated models...\n"

  # Download image-to-video models (token-gated).
  check_hf_auth
  hf download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --include "*.fp16.safetensors" --include "*.json" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
}

# Download all models.
function download_all_models() {
  download_beta_models

  printf "\nDownloading other available models...\n"

  # Download text-to-image and image-to-image models.
  printf "\nDownloading unrestricted models...\n"
  hf download stabilityai/sd-turbo --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download stabilityai/sdxl-turbo --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download runwayml/stable-diffusion-v1-5 --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download prompthero/openjourney-v4 --include "*.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download SG161222/RealVisXL_V4.0 --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download stabilityai/stable-diffusion-3-medium-diffusers --include "*.fp16*.safetensors" --include "*.model" --include "*.json" --include "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  hf download stabilityai/stable-diffusion-3.5-medium --include "transformer/*.safetensors" --include "*model.fp16*" --include "*.model" --include "*.json" --include "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  hf download stabilityai/stable-diffusion-3.5-large --include "transformer/*.safetensors" --include "*model.fp16*" --include "*.model" --include "*.json" --include "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  hf download SG161222/Realistic_Vision_V6.0_B1_noVAE --include "*.fp16.safetensors" --include "*.json" --include "*.txt" --include "*.bin" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models
  hf download black-forest-labs/FLUX.1-schnell --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models

  # Download image-to-video models.
  hf download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" --include "*.json" --cache-dir models

  # Download image-to-text models.
  hf download Salesforce/blip-image-captioning-large --include "*.safetensors" --include "*.json" --cache-dir models

  # Custom pipeline models.
  hf download facebook/sam2-hiera-large --include "*.pt" --include "*.yaml" --cache-dir models

  download_live_models
}

# Download models only for the live-video-to-video pipeline.
function download_live_models() {
  # Check PIPELINE environment variable and download accordingly
  case "$PIPELINE" in
  "streamdiffusion")
    printf "\nPreparing StreamDiffusion live models only...\n"
    prepare_streamdiffusion_models
    ;;
  "comfyui")
    printf "\nDownloading ComfyUI live models only...\n"
    download_comfyui_live_models
    ;;
  "scope")
    printf "\nPreparing Scope live models only...\n"
    prepare_scope_models
    ;;
  "all")
    printf "\Preparing all live models...\n"
    prepare_streamdiffusion_models
    download_comfyui_live_models
    prepare_scope_models
    ;;
  *)
    printf "ERROR: Invalid PIPELINE value: %s. Valid values are: streamdiffusion, comfyui, scope, all\n" "$PIPELINE"
    exit 1
    ;;
  esac
}

function run_pipeline_prepare() {
  local pipeline="$1"
  local image="$2"

  local label="Pipeline-Prepare"
  if [[ "$(docker ps -a -q --filter="label=${label}")" ]]; then
    printf "Previous prepare container run hasn't finished correctly. There are containers still running:\n"
    docker ps -a --filter="label=${label}"
    exit 1
  fi

  if [ "$PULL_IMAGES" = true ]; then
    docker pull "$image"
  fi

  # ai-worker has live-app tags hardcoded in `var livePipelineToImage` so we need to use the same tag in here
  docker image tag "$image" "livepeer/ai-runner:live-app-$pipeline"

  docker run --rm --name "ai-runner-${pipeline}-prepare" -v ./models:/models "${docker_run_flags[@]}" \
    -l "$label" \
    -e HF_HUB_OFFLINE=0 \
    -e HF_HUB_ENABLE_HF_TRANSFER \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$image" bash -c "set -euo pipefail && \
      $CONDA_PYTHON -m pip install --no-cache-dir hf_transfer==0.1.4 && \
      $CONDA_PYTHON -m app.tools.prepare_models --pipeline ${pipeline} && \
      chown -R $(id -u):$(id -g) /models"
}

function prepare_streamdiffusion_models() {
  printf "\nPreparing StreamDiffusion live models...\n"
  run_pipeline_prepare "streamdiffusion" "$AI_RUNNER_STREAMDIFFUSION_IMAGE"
}

function download_comfyui_live_models() {
  printf "\nDownloading ComfyUI live models...\n"

  # ComfyUI
  if [ "$PULL_IMAGES" = true ]; then
    docker pull "${AI_RUNNER_COMFYUI_IMAGE}"
  fi

  # ComfyUI models
  if ! docker image inspect $AI_RUNNER_COMFYUI_IMAGE >/dev/null 2>&1; then
    echo "ERROR: ComfyUI base image $AI_RUNNER_COMFYUI_IMAGE not found"
    exit 1
  fi
  # ai-worker has tags hardcoded in `var livePipelineToImage` so we need to use the same tag in here:
  docker image tag $AI_RUNNER_COMFYUI_IMAGE livepeer/ai-runner:live-app-comfyui
  docker run --rm -v ./models:/models "${docker_run_flags[@]}" -l ComfyUI-Setup-Models $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "cd /workspace/comfystream && \
                 $CONDA_PYTHON src/comfystream/scripts/setup_models.py --workspace /workspace/ComfyUI && \
                 chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI setup_models.py"
      exit 1
    )
}

function build_tensorrt_models() {
  download_live_models

  if [[ "$(docker ps -a -q --filter="label=TensorRT-engines")" ]]; then
    printf "Previous tensorrt run hasn't finished correctly. There are containers still running:\n"
    docker ps -a --filter="label=TensorRT-engines"
    exit 1
  fi
  printf "\nBuilding TensorRT models...\n"

  # Check PIPELINE environment variable and build accordingly
  case "$PIPELINE" in
  "streamdiffusion")
    printf "\nStreamDiffusion models already built on prepare...\n"
    ;;
  "scope")
    printf "\nScope models already built on prepare...\n"
    ;;
  "comfyui")
    printf "\nBuilding ComfyUI TensorRT models only...\n"
    build_comfyui_tensorrt
    ;;
  "scope")
    printf "\nPreparing Scope models only...\n"
    prepare_scope_models
    ;;
  "all")
    printf "\nBuilding all TensorRT models...\n"
    build_comfyui_tensorrt
    prepare_scope_models
    ;;
  *)
    printf "ERROR: Invalid PIPELINE value: %s. Valid values are: streamdiffusion, comfyui, scope, all\n" "$PIPELINE"
    exit 1
    ;;
  esac
}

function prepare_scope_models() {
  printf "\nPreparing Scope models...\n"
  run_pipeline_prepare "scope" "$AI_RUNNER_SCOPE_IMAGE"
}

function prepare_scope_models() {
  printf "\nPreparing Scope models...\n"

  if [ "$PULL_IMAGES" = true ]; then
    docker pull $AI_RUNNER_SCOPE_IMAGE
  fi

  # ai-worker references the live-app tag, so replicate it locally for consistency.
  docker image tag $AI_RUNNER_SCOPE_IMAGE livepeer/ai-runner:live-app-scope

  docker run --rm -v ./models:/models "${docker_run_flags[@]}" \
    -l Scope-Prepare-Models -e HF_HUB_OFFLINE=0 \
    --name scope-prepare-models $AI_RUNNER_SCOPE_IMAGE \
    bash -c "$CONDA_PYTHON -m app.tools.scope.prepare_models --models-dir /models && \
             chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed Scope model preparation"
      exit 1
    )
}

function build_comfyui_tensorrt() {
  printf "\nBuilding ComfyUI TensorRT models...\n"

  # Depth-Anything-Tensorrt
  docker run --rm -v ./models:/models "${docker_run_flags[@]}" -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "cd /workspace/ComfyUI/models/tensorrt/depth-anything && \
                $CONDA_PYTHON /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py --trt-path=./depth_anything_v2_vitl-fp16.engine --onnx-path=./depth_anything_v2_vitl.onnx && \
                $CONDA_PYTHON /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py --trt-path=./depth_anything_vitl14-fp16.engine --onnx-path=./depth_anything_vitl14.onnx && \
                chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI Depth-Anything-Tensorrt"
      exit 1
    )

  # Dreamshaper-8-Dmd-1kstep
  docker run --rm -v ./models:/models "${docker_run_flags[@]}" -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "cd /workspace/comfystream/src/comfystream/scripts && \
                $CONDA_PYTHON ./build_trt.py \
                --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors \
                --out-engine /workspace/ComfyUI/output/tensorrt/static-dreamshaper8_SD15_\\\$stat-b-1-h-512-w-512_00001_.engine && \
                 chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI build_trt.py"
      exit 1
    )

  # Dreamshaper-8-Dmd-1kstep static dynamic 488x704
  docker run --rm -v ./models:/models "${docker_run_flags[@]}" -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "cd /workspace/comfystream/src/comfystream/scripts && \
                $CONDA_PYTHON ./build_trt.py \
                --model /workspace/ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors \
                --out-engine /workspace/ComfyUI/output/tensorrt/dynamic-dreamshaper8_SD15_\$dyn-b-1-4-2-h-448-704-512-w-448-704-512_00001_.engine \
                --width 512 \
                --height 512 \
                --min-width 448 \
                --min-height 448 \
                --max-width 704 \
                --max-height 704 && \
                chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI build_trt.py dynamic engine"
      exit 1
    )

  # FasterLivePortrait
  FASTERLIVEPORTRAIT_DIR="/workspace/ComfyUI/models/liveportrait_onnx"
  docker run --rm -v ./models:/models "${docker_run_flags[@]}" -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "conda run -n comfystream --no-capture-output /workspace/ComfyUI/custom_nodes/ComfyUI-FasterLivePortrait/scripts/build_fasterliveportrait_trt.sh \
             $FASTERLIVEPORTRAIT_DIR $FASTERLIVEPORTRAIT_DIR $FASTERLIVEPORTRAIT_DIR && \
                chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI FasterLivePortrait Tensorrt Engines"
      exit 1
    )
}

# Download models with a restrictive license.
function download_restricted_models() {
  printf "\nDownloading restricted models...\n"

  # Download text-to-image and image-to-image models.
  hf download black-forest-labs/FLUX.1-dev --include "*.safetensors" --include "*.json" --include "*.txt" --include "*.model" --exclude ".onnx" --exclude ".onnx_data" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  # Download LLM models (Warning: large model size)
  hf download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.json" --include "*.bin" --include "*.safetensors" --include "*.txt" --cache-dir models

}

function download_batch_models() {
  printf "\nDownloading Batch models...\n"

  hf download facebook/sam2-hiera-large --include "*.pt" --include "*.yaml" --cache-dir models
}

# Enable XET High Performance.
# See: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables#hfxethighperformance
export HF_XET_HIGH_PERFORMANCE=1

# Use HF_TOKEN if set, otherwise use Hugging Face CLI's login.
[ -n "$HF_TOKEN" ] && TOKEN_FLAG="--token=${HF_TOKEN}" || TOKEN_FLAG=""

# Parse command-line arguments.
MODE="all"
for arg in "$@"; do
  case $arg in
  --beta)
    MODE="beta"
    shift
    ;;
  --restricted)
    MODE="restricted"
    shift
    ;;
  --live)
    MODE="live"
    shift
    ;;
  --tensorrt)
    MODE="tensorrt"
    shift
    ;;
  --batch)
    MODE="batch"
    shift
    ;;
  --help)
    display_help
    exit 0
    ;;
  --select-gpu)
    GPU_SELECT=1
    shift
    ;;
  *)
    shift
    ;;
  esac
done

echo "Starting livepeer AI subnet model downloader..."
echo "Creating 'models' directory in the current working directory..."
mkdir -p models/checkpoints models/StreamDiffusion--engines models/insightface models/StreamDiffusion--models models/ComfyUI--{models,output}

echo "Checking if 'hf' Hugging Face CLI is installed..."
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: The Hugging Face CLI is required to download models. Please install it using 'pip install huggingface_hub'."
  exit 1
fi

if [ "$GPU_SELECT" == 1 ]; then
  select_gpu && export docker_run_flags=(--runtime nvidia -e NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES")
else
  export docker_run_flags=(--gpus all)
fi

if [ "$MODE" = "beta" ]; then
  download_beta_models
elif [ "$MODE" = "restricted" ]; then
  download_restricted_models
elif [ "$MODE" = "live" ]; then
  download_live_models
elif [ "$MODE" = "tensorrt" ]; then
  build_tensorrt_models
elif [ "$MODE" = "batch" ]; then
  download_batch_models
else
  download_all_models
fi

printf "\nAll models downloaded successfully!\n"

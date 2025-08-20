#!/bin/bash

set -e
[ -v DEBUG ] && set -x

# ComfyUI image configuration
PULL_IMAGES=${PULL_IMAGES:-true}
AI_RUNNER_COMFYUI_IMAGE=${AI_RUNNER_COMFYUI_IMAGE:-livepeer/ai-runner:live-app-comfyui}
AI_RUNNER_STREAMDIFFUSION_IMAGE=${AI_RUNNER_STREAMDIFFUSION_IMAGE:-livepeer/ai-runner:live-app-streamdiffusion}
CONDA_PYTHON="/workspace/miniconda3/envs/comfystream/bin/python"
PIPELINE=${PIPELINE:-all}

# Checks HF_TOKEN and huggingface-cli login status and throw warning if not authenticated.
check_hf_auth() {
  if [ -z "$HF_TOKEN" ] && [ "$(huggingface-cli whoami)" = "Not logged in" ]; then
    printf "WARN: Not logged in and HF_TOKEN not set. Log in with 'huggingface-cli login' or set HF_TOKEN to download token-gated models.\n"
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
  echo "  PIPELINE  When using --live or --tensorrt, specify which pipeline to use: 'streamdiffusion', 'comfyui', or 'all' (default)"
  echo "  HF_TOKEN  HuggingFace token for downloading token-gated models"
  echo "  DEBUG  Enable debug mode with set -x"
}

# Download recommended models during beta phase.
function download_beta_models() {
  printf "\nDownloading recommended beta phase models...\n"

  printf "\nDownloading unrestricted models...\n"

  # Download text-to-image and image-to-image models.
  huggingface-cli download SG161222/RealVisXL_V4.0_Lightning --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download ByteDance/SDXL-Lightning --include "*unet.safetensors" --cache-dir models
  huggingface-cli download timbrooks/instruct-pix2pix --include "*.fp16.safetensors" "*.json" "*.txt" --cache-dir models

  # Download upscale models
  huggingface-cli download stabilityai/stable-diffusion-x4-upscaler --include "*.fp16.safetensors" --cache-dir models

  # Download audio-to-text models.
  huggingface-cli download openai/whisper-large-v3 --include "*.safetensors" "*.json" --cache-dir models
  huggingface-cli download distil-whisper/distil-large-v3 --include "*.safetensors" "*.json" --cache-dir models
  huggingface-cli download openai/whisper-medium --include "*.safetensors" "*.json" --cache-dir models

  # Download custom pipeline models.
  huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models
  huggingface-cli download parler-tts/parler-tts-large-v1 --include "*.safetensors" "*.json" "*.model" --cache-dir models

  printf "\nDownloading token-gated models...\n"

  # Download image-to-video models (token-gated).
  check_hf_auth
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --include "*.fp16.safetensors" "*.json" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
}

# Download all models.
function download_all_models() {
  download_beta_models

  printf "\nDownloading other available models...\n"

  # Download text-to-image and image-to-image models.
  printf "\nDownloading unrestricted models...\n"
  huggingface-cli download stabilityai/sd-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download stabilityai/sdxl-turbo --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download runwayml/stable-diffusion-v1-5 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download prompthero/openjourney-v4 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download SG161222/RealVisXL_V4.0 --include "*.fp16.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --include "*.fp16*.safetensors" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  huggingface-cli download stabilityai/stable-diffusion-3.5-medium --include "transformer/*.safetensors" "*model.fp16*" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  huggingface-cli download stabilityai/stable-diffusion-3.5-large --include "transformer/*.safetensors" "*model.fp16*" "*.model" "*.json" "*.txt" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  huggingface-cli download SG161222/Realistic_Vision_V6.0_B1_noVAE --include "*.fp16.safetensors" "*.json" "*.txt" "*.bin" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download black-forest-labs/FLUX.1-schnell --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude ".onnx" ".onnx_data" --cache-dir models

  # Download image-to-video models.
  huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "*.fp16.safetensors" "*.json" --cache-dir models

  # Download image-to-text models.
  huggingface-cli download Salesforce/blip-image-captioning-large --include "*.safetensors" "*.json" --cache-dir models

  # Custom pipeline models.
  huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models

  download_live_models
}

# Download models only for the live-video-to-video pipeline.
function download_live_models() {
  # Check PIPELINE environment variable and download accordingly
  case "$PIPELINE" in
  "streamdiffusion")
    printf "\nDownloading StreamDiffusion live models only...\n"
    download_streamdiffusion_live_models
    ;;
  "comfyui")
    printf "\nDownloading ComfyUI live models only...\n"
    download_comfyui_live_models
    ;;
  "all")
    printf "\nDownloading all live models...\n"
    download_streamdiffusion_live_models
    download_comfyui_live_models
    ;;
  *)
    printf "ERROR: Invalid PIPELINE value: %s. Valid values are: streamdiffusion, comfyui, all\n" "$PIPELINE"
    exit 1
    ;;
  esac
}

function download_streamdiffusion_live_models() {
  printf "\nDownloading StreamDiffusion live models...\n"

  # StreamDiffusion
  huggingface-cli download stabilityai/sd-turbo --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download varb15/PerfectPhotonV2.1 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models

  # SD2.1 (turbo) ControlNet models
  huggingface-cli download thibaud/controlnet-sd21-openpose-diffusers --include "*.bin" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download thibaud/controlnet-sd21-hed-diffusers --include "*.bin" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download thibaud/controlnet-sd21-canny-diffusers --include "*.bin" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download thibaud/controlnet-sd21-depth-diffusers --include "*.bin" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download thibaud/controlnet-sd21-color-diffusers --include "*.bin" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  # SD1.5 controlnet models
  huggingface-cli download lllyasviel/control_v11f1p_sd15_depth --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
  huggingface-cli download lllyasviel/control_v11f1e_sd15_tile --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models

  # Preprocessor models
  huggingface-cli download yuvraj108c/Depth-Anything-2-Onnx --include "depth_anything_v2_vits.onnx" --cache-dir models
  huggingface-cli download yuvraj108c/yolo-nas-pose-onnx --include "yolo_nas_pose_l_0.5.onnx" --cache-dir models
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
  docker run --rm -v ./models:/models --gpus all -l ComfyUI-Setup-Models $AI_RUNNER_COMFYUI_IMAGE \
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
    printf "\nBuilding StreamDiffusion TensorRT models only...\n"
    build_streamdiffusion_tensorrt
    ;;
  "comfyui")
    printf "\nBuilding ComfyUI TensorRT models only...\n"
    build_comfyui_tensorrt
    ;;
  "all")
    printf "\nBuilding all TensorRT models...\n"
    build_streamdiffusion_tensorrt
    build_comfyui_tensorrt
    ;;
  *)
    printf "ERROR: Invalid PIPELINE value: %s. Valid values are: streamdiffusion, comfyui, all\n" "$PIPELINE"
    exit 1
    ;;
  esac
}

function build_streamdiffusion_tensorrt() {
  printf "\nBuilding StreamDiffusion TensorRT models...\n"

  # StreamDiffusion (compile a matrix of models and timesteps)
  if [ "$PULL_IMAGES" = true ]; then
    docker pull $AI_RUNNER_STREAMDIFFUSION_IMAGE
  fi
  # ai-worker has tags hardcoded in `var livePipelineToImage` so we need to use the same tag in here:
  docker image tag $AI_RUNNER_STREAMDIFFUSION_IMAGE livepeer/ai-runner:live-app-streamdiffusion

  # SD2.1 (turbo) models
  docker run --rm -v ./models:/models --gpus all \
    -l TensorRT-engines -e HF_HUB_OFFLINE=0 \
    --name streamdiffusion-tensorrt-build $AI_RUNNER_STREAMDIFFUSION_IMAGE \
    bash -c "./app/tools/streamdiffusion/build_tensorrt_internal.sh \
              --models 'stabilityai/sd-turbo' \
              --opt-timesteps '3' \
              --min-timesteps '1' \
              --max-timesteps '4' \
              --controlnets 'thibaud/controlnet-sd21-openpose-diffusers thibaud/controlnet-sd21-hed-diffusers thibaud/controlnet-sd21-canny-diffusers thibaud/controlnet-sd21-depth-diffusers thibaud/controlnet-sd21-color-diffusers' \
              --build-depth-anything \
              --build-pose \
              && \
            chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed streamdiffusion tensorrt"
      exit 1
    )
  # SD1.5 models
  docker run --rm -v ./models:/models --gpus all \
    -l TensorRT-engines -e HF_HUB_OFFLINE=0 \
    --name streamdiffusion-tensorrt-build $AI_RUNNER_STREAMDIFFUSION_IMAGE \
    bash -c "./app/tools/streamdiffusion/build_tensorrt_internal.sh \
              --models 'varb15/PerfectPhotonV2.1' \
              --opt-timesteps '3' \
              --min-timesteps '1' \
              --max-timesteps '4' \
              --controlnets 'lllyasviel/control_v11f1p_sd15_depth lllyasviel/control_v11f1e_sd15_tile' \
              --build-depth-anything \
              --build-pose \
              && \
            chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed streamdiffusion tensorrt"
      exit 1
    )
}

function build_comfyui_tensorrt() {
  printf "\nBuilding ComfyUI TensorRT models...\n"

  # Depth-Anything-Tensorrt
  docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
    bash -c "cd /workspace/ComfyUI/models/tensorrt/depth-anything && \
                $CONDA_PYTHON /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py --trt-path=./depth_anything_v2_vitl-fp16.engine --onnx-path=./depth_anything_v2_vitl.onnx && \
                $CONDA_PYTHON /workspace/ComfyUI/custom_nodes/ComfyUI-Depth-Anything-Tensorrt/export_trt.py --trt-path=./depth_anything_vitl14-fp16.engine --onnx-path=./depth_anything_vitl14.onnx && \
                chown -R $(id -u):$(id -g) /models" ||
    (
      echo "failed ComfyUI Depth-Anything-Tensorrt"
      exit 1
    )

  # Dreamshaper-8-Dmd-1kstep
  docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
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
  docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
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
  docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
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
  huggingface-cli download black-forest-labs/FLUX.1-dev --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude ".onnx" ".onnx_data" --cache-dir models ${TOKEN_FLAG:+"$TOKEN_FLAG"}
  # Download LLM models (Warning: large model size)
  huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "*.json" "*.bin" "*.safetensors" "*.txt" --cache-dir models

}

function download_batch_models() {
  printf "\nDownloading Batch models...\n"

  huggingface-cli download facebook/sam2-hiera-large --include "*.pt" "*.yaml" --cache-dir models
}

# Enable HF transfer acceleration.
# See: https://huggingface.co/docs/huggingface_hub/v0.22.1/package_reference/environment_variables#hfhubenablehftransfer.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use HF_TOKEN if set, otherwise use huggingface-cli's login.
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
  *)
    shift
    ;;
  esac
done

echo "Starting livepeer AI subnet model downloader..."
echo "Creating 'models' directory in the current working directory..."
mkdir -p models/checkpoints models/StreamDiffusion--engines models/ComfyUI--{models,output}

# Ensure 'huggingface-cli' is installed.
echo "Checking if 'huggingface-cli' is installed..."
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "WARN: The huggingface-cli is required to download models. Please install it using 'pip install huggingface_hub[cli,hf_transfer]'."
  exit 1
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

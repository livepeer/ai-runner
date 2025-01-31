#!/bin/bash

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
    huggingface-cli download KBlueLeaf/kohaku-v2.1 --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models
    huggingface-cli download stabilityai/sd-turbo --include "*.safetensors" "*.json" "*.txt" --exclude ".onnx" ".onnx_data" --cache-dir models

    # ComfyUI
    AI_RUNNER_COMFYUI_IMAGE=${AI_RUNNER_COMFYUI_IMAGE:-livepeer/ai-runner:live-app-comfyui}
    # docker pull $AI_RUNNER_COMFYUI_IMAGE
    # ai-worker has tags hardcoded in `var livePipelineToImage` so we need to use the same tag in here:
    docker image tag $AI_RUNNER_COMFYUI_IMAGE livepeer/ai-runner:live-app-comfyui
    docker run --rm -v ./models:/models --gpus all -l ComfyUI-Setup-Models $AI_RUNNER_COMFYUI_IMAGE \
        bash -c "cd /comfystream && \
                 python src/comfystream/scripts/setup_models.py --workspace /ComfyUI && \
                 adduser $(id -u -n) && \
                 chown -R $(id -u -n):$(id -g -n) /models" \
        || (echo "failed ComfyUI setup_models.py"; return 1)
}

function build_tensorrt_models() {
    download_live_models

    if [[ "$( docker ps -a -q --filter="label=TensorRT-engines" )" ]]; then
        printf "Previous tensorrt run hasn't finished correclty. There are containers still running:\n"
        docker ps -a --filter="label=TensorRT-engines"
        exit 1
    fi
    printf "\nBuilding TensorRT models...\n"

    # StreamDiffusion (compile a matrix of models and timesteps)
    MODELS="stabilityai/sd-turbo KBlueLeaf/kohaku-v2.1"
    TIMESTEPS="3 4" # This is basically the supported sizes for the t_index_list
    AI_RUNNER_STREAMDIFFUSION_IMAGE=${AI_RUNNER_STREAMDIFFUSION_IMAGE:-livepeer/ai-runner:live-app-streamdiffusion}
    docker pull $AI_RUNNER_STREAMDIFFUSION_IMAGE
    # ai-worker has tags hardcoded in `var livePipelineToImage` so we need to use the same tag in here:
    docker image tag $AI_RUNNER_STREAMDIFFUSION_IMAGE livepeer/ai-runner:live-app-streamdiffusion
    docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_STREAMDIFFUSION_IMAGE \
        bash -c "for model in $MODELS; do
                    for timestep in $TIMESTEPS; do
                        echo \"Building TensorRT engines for model=\$model timestep=\$timestep...\" && \
                        python app/live/StreamDiffusionWrapper/build_tensorrt.py --model-id \$model --timesteps \$timestep
                    done
                done
                adduser $(id -u -n)
                chown -R $(id -u -n):$(id -g -n) /models
                " \
        || (echo "failed streamdiffusion tensorrt"; return 1)

    # TODO: Remove the script download with curl. It should already come in the base image once eliteprox/comfystream#1 is merged.
    docker run --rm -v ./models:/models --gpus all -l TensorRT-engines $AI_RUNNER_COMFYUI_IMAGE \
        bash -c "cd /comfystream/src/comfystream/scripts && \
                 curl -O https://raw.githubusercontent.com/pschroedl/comfystream/refs/heads/10_29/build_trt/src/comfystream/scripts/build_trt.py && \
                 python ./build_trt.py \
                --model /ComfyUI/models/unet/dreamshaper-8-dmd-1kstep.safetensors \
                --out-engine /ComfyUI/output/tensorrt/static-dreamshaper8_SD15_\$stat-b-1-h-512-w-512_00001_.engine" \
        || (echo "failed ComfyUI build_trt.py"; return 1)
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
for arg in "$@"
do
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
mkdir -p models
mkdir -p models/checkpoints
mkdir -p models/StreamDiffusion--engines models/ComfyUI--models models/ComfyUI--output

# Ensure 'huggingface-cli' is installed.
echo "Checking if 'huggingface-cli' is installed..."
if ! command -v huggingface-cli > /dev/null 2>&1; then
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

#!/bin/bash
set -e  # Exit on any error

# Configuration
export PIPELINE=comfyui
APP_IMAGE="livepeer/ai-runner:live-app-${PIPELINE}"

# Build application image (final image for comfyui)
build_app() {
    echo "Building application image..."
    # Note: comfyui uses an external base image, so no live-base dependency
    docker build -t ${APP_IMAGE} -f docker/Dockerfile.live-app-comfyui ..
}

# Download checkpoints
download_checkpoints() {
    echo "Downloading checkpoints..."
    ./dl_checkpoints.sh --live
    ./dl_checkpoints.sh --tensorrt
}

# Run the application
run_app() {
    echo "Running the application..."
    docker run -it --rm \
        --name video-to-video \
        --gpus all \
        -p 8000:8000 \
        -v ./models:/models \
        -e PIPELINE=live-video-to-video \
        -e MODEL_ID=comfyui \
        ${APP_IMAGE}
}

# Target: build-all
build_all() {
    build_app
    download_checkpoints
}

# Parse command line arguments
case "$1" in
    "app")
        build_app
        ;;
    "models")
        download_checkpoints
        ;;
    "run")
        run_app
        ;;
    "all")
        build_all
        ;;
    *)
        echo "Usage: $0 {app|models|run|all}"
        echo "  app        - Build application image"
        echo "  models     - Download model checkpoints"
        echo "  run        - Run the application"
        echo "  all        - Build everything in sequence"
        exit 1
        ;;
esac

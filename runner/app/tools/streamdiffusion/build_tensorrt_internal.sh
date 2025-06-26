#!/bin/bash

# Internal TensorRT engine build script that runs inside the container
# This script is designed to be executed within the AI runner container environment

[ -v DEBUG ] && set -x

# Configuration
CONDA_PYTHON="/workspace/miniconda3/envs/comfystream/bin/python"
MODELS="stabilityai/sd-turbo KBlueLeaf/kohaku-v2.1"
TIMESTEPS="3 4" # This is basically the supported sizes for the t_index_list
DIMENSIONS="384x704 704x384"

# Function to display help
function display_help() {
    echo "Description: This script builds TensorRT engines for StreamDiffusion models inside the container."
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --models MODEL_LIST     Comma-separated list of models (default: stabilityai/sd-turbo,KBlueLeaf/kohaku-v2.1)"
    echo "  --timesteps TIMESTEPS   Space-separated list of timesteps (default: 4)"
    echo "  --dimensions DIMS       Space-separated list of dimensions in WxH format (default: 384x704 704x384)"
    echo "  --output-dir DIR        Output directory for TensorRT engines (default: engines)"
    echo "  --help                  Display this help message"
}

# Default values
OUTPUT_DIR="./engines"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="${2//,/ }"  # Convert comma-separated to space-separated
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --dimensions)
            DIMENSIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Ensure we're in the right environment
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "ERROR: Python executable not found at $CONDA_PYTHON"
    echo "This script must be run inside the AI runner container environment."
    exit 1
fi

# Check if the build script exists
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
BUILD_SCRIPT="$SCRIPT_DIR/build_tensorrt.py"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: Build script not found at $BUILD_SCRIPT"
    echo "Make sure the StreamDiffusion build script is available in the same directory as this script."
    exit 1
fi

# Create output directory
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo
fi

# Build TensorRT engines
echo "Starting TensorRT engine build process..."
echo "Models: $MODELS"
echo "Timesteps: $TIMESTEPS"
echo "Dimensions: $DIMENSIONS"
echo "Output directory: $OUTPUT_DIR"
echo

total_builds=0

# Calculate total number of builds
for model in $MODELS; do
    for timestep in $TIMESTEPS; do
        for dim in $DIMENSIONS; do
            ((total_builds++))
        done
    done
done

echo "Total builds to perform: $total_builds"
echo

current_build=0

for model in $MODELS; do
    for timestep in $TIMESTEPS; do
        for dim in $DIMENSIONS; do
            ((current_build++))

            # Parse dimensions
            width=${dim%x*}
            height=${dim#*x}

            echo "[$current_build/$total_builds] Building TensorRT engine for:"
            echo "  Model: $model"
            echo "  Timesteps: $timestep"
            echo "  Dimensions: ${width}x${height}"

            # Build the engine
            if $CONDA_PYTHON "$BUILD_SCRIPT" \
                --model-id "$model" \
                --timesteps "$timestep" \
                --width "$width" \
                --height "$height" \
                --engine-dir "$OUTPUT_DIR"; then
                echo "  ✓ Success"
            else
                echo "  ✗ Failed"
                echo "Aborting build process due to failure."
                exit 1
            fi
            echo
        done
    done
done

# Summary
echo "Build process completed!"
echo "Total builds: $total_builds"

# List built engines
if [ -d "$OUTPUT_DIR" ]; then
    echo
    echo "Built engines:"
    find "$OUTPUT_DIR" -name "*.engine" -o -name "*.trt" 2>/dev/null | sort
fi

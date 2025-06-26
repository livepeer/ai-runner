#!/usr/bin/env python3
"""
Debug script to analyze saved tensors from StreamDiffusion pipeline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEBUG_LOG_DIR = "/home/user/workspace/ai-runner/runner/models/debug-logs"

def analyze_tensor_file(filepath: str):
    """Analyze a single tensor file"""
    try:
        tensor = np.load(filepath)
        print(f"\nFile: {os.path.basename(filepath)}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Min: {tensor.min()}")
        print(f"  Max: {tensor.max()}")
        print(f"  Mean: {tensor.mean():.3f}")
        print(f"  Std: {tensor.std():.3f}")

        # Check if it's mostly black
        if tensor.max() < 10:  # Very low values
            print(f"  ⚠️  WARNING: Tensor appears to be mostly black (max < 10)")
        elif tensor.max() < 50:  # Low values
            print(f"  ⚠️  WARNING: Tensor has low values (max < 50)")

        return tensor
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def visualize_tensor(tensor: np.ndarray, title: str, save_path: str = None):
    """Visualize a tensor as an image"""
    if tensor is None:
        return

    plt.figure(figsize=(10, 8))

    if tensor.shape[-1] == 3:  # RGB
        plt.imshow(tensor)
        plt.title(f"{title} (RGB)")
    elif tensor.shape[-1] == 1:  # Grayscale
        plt.imshow(tensor.squeeze(), cmap='gray')
        plt.title(f"{title} (Grayscale)")
    else:
        # Show first channel
        plt.imshow(tensor[:, :, 0], cmap='gray')
        plt.title(f"{title} (Channel 0)")

    plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()

def main():
    if not os.path.exists(DEBUG_LOG_DIR):
        print(f"Debug log directory not found: {DEBUG_LOG_DIR}")
        return

    print("=== StreamDiffusion Tensor Debug Analysis ===")
    print(f"Debug log directory: {DEBUG_LOG_DIR}")

    # Find all tensor files
    tensor_files = list(Path(DEBUG_LOG_DIR).glob("*.npy"))
    tensor_files.sort()

    if not tensor_files:
        print("No tensor files found!")
        return

    print(f"\nFound {len(tensor_files)} tensor files:")

    # Group files by frame
    frames = {}
    for filepath in tensor_files:
        filename = filepath.stem
        parts = filename.split('_')
        if len(parts) >= 2:
            frame_key = f"{parts[0]}_{parts[1]}"
            step_name = '_'.join(parts[2:])
            if frame_key not in frames:
                frames[frame_key] = {}
            frames[frame_key][step_name] = str(filepath)

    # Analyze each frame
    for frame_key, steps in frames.items():
        print(f"\n{'='*50}")
        print(f"Frame: {frame_key}")
        print(f"{'='*50}")

        # Analyze each step in order
        step_order = ['input', 'after_permute', 'after_denormalize', 'after_preprocess',
                     'warmup_1', 'warmup_2', 'warmup_3', 'raw_output', 'final_output']

        tensors = {}
        for step in step_order:
            if step in steps:
                tensor = analyze_tensor_file(steps[step])
                tensors[step] = tensor

        # Create visualization for this frame
        if tensors:
            viz_dir = os.path.join(DEBUG_LOG_DIR, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            for step, tensor in tensors.items():
                if tensor is not None:
                    viz_path = os.path.join(viz_dir, f"{frame_key}_{step}.png")
                    visualize_tensor(tensor, f"{frame_key} - {step}", viz_path)

    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"Check the visualizations directory: {os.path.join(DEBUG_LOG_DIR, 'visualizations')}")

if __name__ == "__main__":
    main()

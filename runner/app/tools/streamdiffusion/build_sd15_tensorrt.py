#!/usr/bin/env python3
"""
Simple hardcoded script to build TensorRT engines for stable-diffusion-v1-5.
This script uses the StreamDiffusion wrapper directly without pipeline abstractions.
"""

import sys
import torch
from streamdiffusion import StreamDiffusionWrapper

import logging

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Enable verbose logging for StreamDiffusion
logging.getLogger("streamdiffusion").setLevel(logging.DEBUG)

def main():
    # Hardcoded configuration for stable-diffusion-v1-5
    # MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    MODEL_ID = "varb15/PerfectPhotonV2.1"
    T_INDEX_LIST = [16, 32, 45]  # Hardcoded timesteps for SD 1.5
    WIDTH = 512
    HEIGHT = 512
    ENGINE_DIR = "engines"

    print(f"Building TensorRT engines for model: {MODEL_ID}")
    print(f"Using timesteps: {T_INDEX_LIST}")
    print(f"Image dimensions: {WIDTH}x{HEIGHT}")

    # Calculate latent dimensions (VAE downscales by factor of 8)
    latent_width = WIDTH // 8
    latent_height = HEIGHT // 8
    print(f"Expected latent dimensions: {latent_width}x{latent_height}")
    print(f"Engines will be saved to: {ENGINE_DIR}")

    # Create StreamDiffusion wrapper with hardcoded parameters
    try:
        print("Initializing StreamDiffusion wrapper...")
        wrapper = StreamDiffusionWrapper(
            model_id_or_path=MODEL_ID,
            t_index_list=T_INDEX_LIST,
            lora_dict=None,
            mode="img2img",
            output_type="pil",
            lcm_lora_id=None,
            vae_id=None,
            device="cuda",
            dtype=torch.float16,
            frame_buffer_size=1,
            width=WIDTH,
            height=HEIGHT,
            warmup=10,
            acceleration="tensorrt",
            do_add_noise=True,
            device_ids=None,
            use_lcm_lora=True,
            use_tiny_vae=True,
            enable_similar_image_filter=False,
            similar_image_filter_threshold=0.98,
            similar_image_filter_max_skip_frame=10,
            use_denoising_batch=True,
            cfg_type="self",
            seed=2,
            use_safety_checker=False,
            engine_dir=ENGINE_DIR,
            build_engines_if_missing=True,
            use_controlnet=False,
            controlnet_config=None,
        )

        print("Preparing wrapper with dummy prompt to trigger TensorRT engine building...")
        # Call prepare with a simple prompt to trigger engine building
        wrapper.prepare(
            prompt="a beautiful landscape",
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=1.2,
            delta=1.0,
        )

        print("TensorRT engine building completed successfully!")

    except Exception as e:
        print(f"ERROR: Failed to build TensorRT engines: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

import argparse
from typing import Dict, List, Any

from streamdiffusion import StreamDiffusionWrapper

def create_controlnet_configs(controlnet_model_ids: List[str], image_resolution: int = 512) -> List[Dict[str, Any]]:
    """Create dummy ControlNet configurations for compilation"""
    controlnet_configs = []

    for model_id in controlnet_model_ids:
        # Create a basic configuration for each ControlNet
        # The exact parameters don't matter for compilation, just need the model to be loaded
        config = {
            'model_id': model_id,
            'conditioning_scale': 0.5,  # Default scale
            'preprocessor': "passthrough",  # Simple preprocessor
            'preprocessor_params': {"image_resolution": image_resolution},
            'enabled': True,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        controlnet_configs.append(config)

    return controlnet_configs

def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for StreamDiffusion")

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID or path to load (e.g. KBlueLeaf/kohaku-v2.1, stabilityai/sd-turbo)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=3,
        help="Number of timesteps in t_index_list (default: 3)"
    )

    parser.add_argument(
        "--engine-dir",
        type=str,
        default="engines",
        help="Directory to save TensorRT engines (default: engines)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the output image (default: 512)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the output image (default: 512)"
    )

    parser.add_argument(
        "--controlnets",
        type=str,
        default="",
        help="Space-separated list of ControlNet model IDs to compile (e.g. 'lllyasviel/control_v11f1e_sd15_tile lllyasviel/control_v11f1p_sd15_depth')"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Create t_index_list based on number of timesteps. Only the size matters...
    t_index_list = list(range(0, 50, 50 // args.timesteps))[:args.timesteps]

    print(f"Building TensorRT engines for model: {args.model_id}")
    print(f"Using {args.timesteps} timesteps: {t_index_list}")
    print(f"Image dimensions: {args.width}x{args.height}")

    # Calculate latent dimensions (VAE downscales by factor of 8)
    latent_width = args.width // 8
    latent_height = args.height // 8
    print(f"Expected latent dimensions: {latent_width}x{latent_height}")
    print(f"Engines will be saved to: {args.engine_dir}")

    # Create ControlNet configurations if provided
    controlnet_config = None
    use_controlnet = False
    if args.controlnets:
        controlnet_model_ids = args.controlnets.split()
        controlnet_config = create_controlnet_configs(controlnet_model_ids, max(args.width, args.height))
        use_controlnet = True
        print(f"ControlNets ({len(controlnet_model_ids)}):")
        for i, cn_id in enumerate(controlnet_model_ids):
            print(f"  {i}: {cn_id}")

    # Initialize wrapper which will trigger TensorRT engine building
    wrapper = StreamDiffusionWrapper(
        mode="img2img",
        acceleration="tensorrt",
        frame_buffer_size=1,
        model_id_or_path=args.model_id,
        t_index_list=t_index_list,
        engine_dir=args.engine_dir,
        width=args.width,
        height=args.height,
        build_engines_if_missing=True,  # Explicitly enable engine building for this script
        use_controlnet=use_controlnet,
        controlnet_config=controlnet_config,
    )

    print("TensorRT engine building completed successfully!")

if __name__ == "__main__":
    main()

import argparse
import os
import sys

# Add the current script's directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wrapper import StreamDiffusionWrapper

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

    return parser.parse_args()

def main():
    args = parse_args()

    # Create t_index_list based on number of timesteps. Only the size matters...
    t_index_list = list(range(0, 50, 50 // args.timesteps))[:args.timesteps]

    print(f"Building TensorRT engines for model: {args.model_id}")
    print(f"Using {args.timesteps} timesteps: {t_index_list}")
    print(f"Engines will be saved to: {args.engine_dir}")

    # Initialize wrapper which will trigger already TensorRT engine building
    wrapper = StreamDiffusionWrapper(
        mode="img2img",
        acceleration="tensorrt",
        frame_buffer_size=1,
        model_id_or_path=args.model_id,
        t_index_list=t_index_list,
        engine_dir=args.engine_dir
    )

    print("TensorRT engine building completed successfully!")

if __name__ == "__main__":
    main()

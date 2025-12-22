"""This script benchmarks GPU memory usage and inference time for various AI pipelines.
"""

import argparse
import os
from pathlib import Path
from time import time
from typing import Any, List

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from runner.app import load_pipeline
from runner.pipelines.base import Pipeline

CURRENT_DIR = Path(__file__).parent

PROMPT = "a mountain lion"
EXAMPLE_IMAGE_PATH = Path(CURRENT_DIR, "example_data/image.png")
EXAMPLE_IMAGE_LOW_RES_PATH = Path(CURRENT_DIR, "example_data/image-low-res.png")
EXAMPLE_AUDIO_FILE_PATH = Path(CURRENT_DIR, "example_data/test_audio.flac")


def create_upload_file(file_path: Path) -> UploadFile:
    """Creates an UploadFile object from a file path.

    Args:
        file_path: The path to the file.

    Returns:
        UploadFile: The UploadFile object.
    """
    try:
        return UploadFile(filename=file_path.name, file=open(file_path, "rb"))
    except IOError as e:
        print(f"Error opening file {file_path}: {e}")
        raise


def get_example_input(pipeline_name: str) -> dict:
    """Returns example input for the specified pipeline.

    Args:
        pipeline_name: The name of the pipeline.

    Returns:
        dict: A dictionary containing example input for the specified pipeline.

    Raises:
        NotImplementedError: If example input is not implemented for the specified
            pipeline.
    """
    try:
        image_path = (
            EXAMPLE_IMAGE_LOW_RES_PATH
            if "UpscalePipeline" in pipeline_name
            else EXAMPLE_IMAGE_PATH
        )
        example_image = Image.open(image_path).convert("RGB")
    except IOError as e:
        print(f"Error opening image file {EXAMPLE_IMAGE_PATH}: {e}")
        raise

    example_inputs = {
        "AudioToTextPipeline": {"audio": create_upload_file(EXAMPLE_AUDIO_FILE_PATH)},
        "TextToImagePipeline": {"prompt": PROMPT},
        "ImageToImagePipeline": {"prompt": PROMPT, "image": example_image},
        "ImageToVideoPipeline": {"image": example_image},
        "UpscalePipeline": {"prompt": PROMPT, "image": example_image},
        "SegmentAnything2Pipeline": {"image": example_image},
    }

    if pipeline_name not in example_inputs:
        raise NotImplementedError(
            f"Example input not implemented for this pipeline: {pipeline_name}"
        )

    return example_inputs[pipeline_name]


class BenchMetrics(BaseModel):
    """A class to store benchmarking metrics."""

    inference_time: float
    max_mem_allocated: float
    max_mem_reserved: float


def call_pipeline(pipeline: Pipeline, **kwargs) -> List[Any]:
    """Calls a pipeline with example inputs.

    Args:
        pipeline: The pipeline to call.
        **kwargs: Additional keyword arguments to pass to the pipeline.

    Returns:
        List: The output of the pipeline.
    """
    example_kwargs = get_example_input(pipeline.__class__.__name__)
    kwargs.update(example_kwargs)
    return pipeline(**kwargs)


def bench_pipeline(
    pipeline: Pipeline,
    runs: int = 1,
    num_inference_steps: int = None,
) -> BenchMetrics:
    """Benchmarks a pipeline by calling it multiple times and collecting metrics.

    Args:
        pipeline: The pipeline to benchmark.
        runs: The number of times to call the pipeline.
        num_inference_steps: The number of inference steps to run for the pipeline.

    Returns:
        BenchMetrics: The benchmarking metrics.
    """
    inference_time = np.zeros(runs)
    max_mem_allocated = np.zeros(runs)
    max_mem_reserved = np.zeros(runs)

    kwargs = (
        {"num_inference_steps": num_inference_steps}
        if num_inference_steps is not None
        else {}
    )

    for i in range(runs):
        start = time()
        output = call_pipeline(pipeline, **kwargs)
        if isinstance(output, tuple):
            output = output[0]

        inference_time[i] = time() - start
        max_mem_allocated[i] = torch.cuda.max_memory_allocated() / 1024**3
        max_mem_reserved[i] = torch.cuda.max_memory_reserved() / 1024**3

        print(f"inference {i+1} time: {inference_time[i]:.3f}s")
        print(
            f"inference {i+1} max GPU memory allocated: {max_mem_allocated[i]:.3f}GiB"
        )
        print(f"inference {i+1} max GPU memory reserved: {max_mem_reserved[i]:.3f}GiB")

    return BenchMetrics(
        inference_time=inference_time.mean(),
        max_mem_allocated=max_mem_allocated.mean(),
        max_mem_reserved=max_mem_reserved.mean(),
    )


def main():
    parser = argparse.ArgumentParser(description="A benchmarking tool for AI pipelines")
    parser.add_argument(
        "--pipeline", type=str, required=True, help="the name of the pipeline"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="the ID of the model to load for the pipeline",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        required=False,
        help="the number of times to call the pipeline",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        required=False,
        help=(
            "the number of inference steps to run for the pipeline. Not all pipelines "
            "support this.",
        ),
    )
    args = parser.parse_args()

    print("Starting benchmark...")
    args_dict = vars(args)
    print_parts = [
        f"{key}={value}" for key, value in args_dict.items() if value is not None
    ]
    print(", ".join(print_parts))

    start = time()
    pipeline = load_pipeline(args.pipeline, args.model_id)

    # Collect pipeline load metrics.
    load_time = time() - start
    load_max_mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
    load_max_mem_reserved = torch.cuda.max_memory_reserved() / 1024**3

    # Collect pipeline warmup metrics if stable-fast is enabled.
    if os.getenv("SFAST", "").strip().lower() == "true":
        warmups = 3
        warmup_metrics = bench_pipeline(pipeline, warmups, args.num_inference_steps)

    # Collect pipeline inference metrics.
    metrics = bench_pipeline(pipeline, args.runs, args.num_inference_steps)

    print("\n")
    print("----AGGREGATE METRICS----")
    print("\n")

    print(f"pipeline load time: {load_time:.3f}s")
    print(f"pipeline load max GPU memory allocated: {load_max_mem_allocated:.3f}GiB")
    print(f"pipeline load max GPU memory reserved: {load_max_mem_reserved:.3f}GiB")

    if os.getenv("SFAST", "").strip().lower() == "true":
        print(f"avg warmup inference time: {warmup_metrics.inference_time:.3f}s")
        print(
            f"avg warmup inference max GPU memory allocated: "
            f"{warmup_metrics.max_mem_allocated:.3f}GiB"
        )
        print(
            f"avg warmup inference max GPU memory reserved: "
            f"{warmup_metrics.max_mem_reserved:.3f}GiB"
        )

    print(f"avg inference time: {metrics.inference_time:.3f}s")
    print(f"avg inference max GPU memory allocated: {metrics.max_mem_allocated:.3f}GiB")
    print(f"avg inference max GPU memory reserved: {metrics.max_mem_reserved:.3f}GiB")


if __name__ == "__main__":
    main()

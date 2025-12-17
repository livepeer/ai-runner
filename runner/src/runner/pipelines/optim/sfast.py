"""This module provides a function to enable StableFast optimization for the pipeline.

For more information, see the DeepCache project on GitHub: https://github.com/chengzeyi/stable-fast
"""  # noqa: E501

import logging

from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig, compile

logger = logging.getLogger(__name__)


def compile_model(pipe):
    """Compile the pipeline with StableFast optimization.

    Args:
        pipe: The pipeline to be optimized.

    Returns:
        The optimized pipeline.
    """
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # NOTE: Disable Triton if kernel generation, compilation, and fine-tuning are slow,
    # especially due to insufficient GPU VRAM or outdated architecture.
    try:
        import xformers  # noqa: F401

        config.enable_xformers = True
    except ImportError:
        logger.info("xformers not installed, skip")
    try:
        import triton  # noqa: F401

        config.enable_triton = True
    except ImportError:
        logger.info("Triton not installed, skip")

    pipe = compile(pipe, config)
    return pipe

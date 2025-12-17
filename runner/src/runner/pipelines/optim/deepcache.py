"""This module provides a function to enable DeepCache optimization for the pipeline.

For more information, see the DeepCache project on GitHub: https://github.com/horseee/DeepCache
"""  # noqa: E501

import logging

from DeepCache import DeepCacheSDHelper

logger = logging.getLogger(__name__)


def enable_deepcache(pipe):
    """Enable DeepCache optimization for the pipeline.

    Args:
        pipe: The pipeline to be optimized.

    Returns:
        The optimized pipeline.
    """
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()

    return pipe

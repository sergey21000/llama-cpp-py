import os
from typing import Sequence


class LlamaEnv:
    """Utilities for managing llama.cpp-related environment variables."""

    LLAMA_ARG_PREFIXES = ['LLAMA_ARG_', 'LLAMA_LOG_']
    LLAMA_MODEL_PREFIXES = (
        'LLAMA_ARG_MODEL',
        'LLAMA_ARG_HF',
        'LLAMA_ARG_MMPROJ',
    )

    @classmethod
    def _clear_vars(cls, prefixes: Sequence[str]) -> int:
        """
        Remove environment variables that start with any of the specified prefixes.

        Iterates through all current environment variables and deletes those
        whose names begin with any prefix from the provided sequence.

        Args:
            prefixes: Sequence of string prefixes to match against variable names.
                     Variables starting with any of these prefixes will be removed.

        Returns:
            int: Number of environment variables successfully removed.
        """
        removed = 0
        for key in os.environ:
            if any(key.startswith(prefix) for prefix in prefixes):
                del os.environ[key]
                removed += 1
        return removed

    @classmethod
    def clear_all_vars(cls) -> int:
        """Remove all environment variables starting with LLAMA_ARG_"""
        return cls._clear_vars(prefixes=cls.LLAMA_ARG_PREFIXES)

    @classmethod
    def clear_model_vars(cls) -> int:
        """
        Remove LLAMA_ARG_ environment variables related to model configuration.
        Clears variables for model path (MODEL), HuggingFace integration (HF),
        and multimodal projections (MMPROJ).
        """
        return cls._clear_vars(prefixes=cls.LLAMA_MODEL_PREFIXES)

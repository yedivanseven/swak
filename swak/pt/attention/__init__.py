"""Attention modules readied for injecting custom positional encodings."""

from .self_gqa import GroupedQuerySelfAttention
from .self_mha import MultiheadedSelfAttention

__all__ = [
    'GroupedQuerySelfAttention',
    'MultiheadedSelfAttention'
]

"""Combine your embedded features into vectors of size model-dimension.

For interpretability, feature importance is available in all mixer flavors.

"""

from .weights import GlobalWeightsMixer, InstanceWeightsMixer
from .attention import CrossAttentionMixer, SelfAttentionMixer

__all__ = [
    'GlobalWeightsMixer',
    'InstanceWeightsMixer',
    'CrossAttentionMixer',
    'SelfAttentionMixer',
]

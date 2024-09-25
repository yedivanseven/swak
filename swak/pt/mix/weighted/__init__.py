"""Combine feature embedding through weighted sums.

Depending on whether these weights are learnable themselves and on whether
they depend on the input features or not, some type of feature importance
can be provided.

"""

from .constant import ConstantSumMixer
from .variable import VariableSumMixer
from .activated import ActivatedSumMixer
from .gated import GatedSumMixer
from .gated_residual import GatedResidualSumMixer

__all__ = [
    'ConstantSumMixer',
    'VariableSumMixer',
    'ActivatedSumMixer',
    'GatedSumMixer',
    'GatedResidualSumMixer'
]

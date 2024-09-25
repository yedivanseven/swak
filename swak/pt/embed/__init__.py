"""Flexibly project your features into embedding space.

The first step in many modern neural-network architectures is to transform
input features into vectors in an embedding space with a certain number of
dimensions, the "model dimension" (or overall "bus width" of the model). This
subpackage provides several ways to do that for both numerical and categorical
features so that, when combined, all are treated on equal footing.

"""

from .activated import ActivatedEmbedder
from .gated import GatedEmbedder
from .gated_residual import GatedResidualEmbedder
from .numerical import NumericalEmbedder
from .categorical import CategoricalEmbedder
from .feature import FeatureEmbedder

__all__ = [
    'ActivatedEmbedder',
    'GatedEmbedder',
    'GatedResidualEmbedder',
    'NumericalEmbedder',
    'CategoricalEmbedder',
    'FeatureEmbedder'
]

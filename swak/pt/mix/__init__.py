"""Combine your embedded features into vectors of size model-dimension.

Two ways of doing this are provided here. One is to form a (weighted) sum
of the feature embeddings. When these weights are learnable themselves, they
can be interpreted as feature importance. The other is to concatenate the
embedding vectors of all features into a single, wide vector and to then
project it down again into a space with the same dimensions of the embedding
space.

"""

from .activated import ActivatedConcatMixer
from .gated import GatedConcatMixer
from .gated_residual import GatedResidualConcatMixer


__all__ = [
    'ActivatedConcatMixer',
    'GatedConcatMixer',
    'GatedResidualConcatMixer'
]

"""Convenient classes (and functions) that do not fit any other category."""

from .identity import identity, Identity
from .delayed import Delayed
from .finalizers import Finalizer, NegativeBinomialFinalizer
from .compile import Compile
from .shape import Cat, Stack, LazyCatDim0
from .configure import ConfigureTorch

__all__ = [
    'identity',
    'Identity',
    'Delayed',
    'Finalizer',
    'NegativeBinomialFinalizer',
    'Compile',
    'Stack',
    'Cat',
    'LazyCatDim0',
    'ConfigureTorch'
]

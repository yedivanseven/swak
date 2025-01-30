from typing import Any, Self
from collections.abc import Callable, Iterator
from abc import ABC, abstractmethod
from pandas import DataFrame
from torch import Tensor, dtype, device
from torch.nn import Module, Dropout, AlphaDropout
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

type Functional = Callable[[Tensor], Tensor]
type Tensors1T = tuple[Tensor]
type Tensors2T = tuple[Tensor, Tensor]
type Tensors3T = tuple[Tensor, Tensor, Tensor]
type Tensors4T = tuple[Tensor, Tensor, Tensor, Tensor]
type Tensors = tuple[Tensor, ...]
type Dtype = dtype
type Device = device
type Drop = Dropout | AlphaDropout
type Batch = tuple[tuple[Any, ...], Tensor]
type Batches = Iterator[Batch]

__all__ = [
    'DataFrame',
    'Tensor',
    'Module',
    'Functional',
    'Tensors1T',
    'Tensors2T',
    'Tensors3T',
    'Tensors4T',
    'Tensors',
    'Dtype',
    'Device',
    'Drop',
    'Optimizer',
    'LRScheduler',
    'Batch',
    'Batches',
    'Resettable',
    'Block'
]


class Resettable(Module, ABC):
    """Abstract base class for Modules with a ``reset_parameters`` method."""

    def __init__(self, *_: Any, **__: Any) -> None:
        super().__init__()

    @abstractmethod
    def reset_parameters(self) -> None:
        """Subclasses implement in-place reset of all internal parameters."""
        ...


class Block(Resettable):
    """Abstract base class for stackable/repeatable neural-network components.

    The input and output tensors of such components must have the same
    dimensions and sizes!

    """

    @abstractmethod
    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        ...

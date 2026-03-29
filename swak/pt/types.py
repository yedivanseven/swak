from typing import Any, Self, overload
from collections.abc import Callable, Iterator
from abc import ABC, abstractmethod
from pandas import DataFrame
from torch import Tensor
from torch.nn import Module, Dropout, AlphaDropout
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

type Functional = Callable[[Tensor], Tensor]
type Tensors1T = tuple[Tensor]
type Tensors2T = tuple[Tensor, Tensor]
type Tensors3T = tuple[Tensor, Tensor, Tensor]
type Tensors4T = tuple[Tensor, Tensor, Tensor, Tensor]
type Tensors = tuple[Tensor, ...]
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
    'Drop',
    'Optimizer',
    'LRScheduler',
    'Batch',
    'Batches',
    'Resettable',
    'Block',
    'Bag',
    'Mixer'
]


class Resettable(Module, ABC):
    """Abstract base class for Modules with a ``reset_parameters`` method."""

    def __init__(self, *_: Any, **__: Any) -> None:
        super().__init__()

    @overload
    @staticmethod
    def _reset(module: Module, device, dtype) -> Module:
        ...

    @overload
    @staticmethod
    def _reset(function: Functional, device, dtype) -> Functional:
        ...

    @staticmethod
    def _reset(obj, device, dtype):
        """Reset parameters of activations if they have any."""
        if isinstance(obj, Module):
            if hasattr(obj, 'reset_parameters'):
                obj.reset_parameters()
            return obj.to(device=device, dtype=dtype)
        return obj

    @abstractmethod
    def reset_parameters(self) -> None:
        """Subclasses implement in-place reset of all internal parameters."""
        ...


class Block(Resettable):
    """Abstract base class for neural-network components."""

    # ToDo: Comment out once everything is a block!
    # @property
    # @abstractmethod
    # def mod_dim(self) -> int:
    #     """Return the embedding dimension of the module."""
    #
    # @property
    # @abstractmethod
    # def device(self) -> torch.device | None:
    #     """Return the device that parameters/weights live on, if possible."""
    #
    # @property
    # @abstractmethod
    # def dtype(self) -> torch.dtype | None:
    #     """Return the dtype of parameters/weight, if possible."""

    @abstractmethod
    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""


class Bag(Block):

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Return the number of features in the bag."""


class Mixer(Block):

    @abstractmethod
    def importance(self, inp: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return per-instance feature importance."""

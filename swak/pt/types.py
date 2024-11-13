from typing import Any
from collections.abc import Callable, Iterator
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
    'Batches'
]

from collections.abc import Callable
from pandas import DataFrame
from torch import Tensor, dtype, device
from torch.nn import Module, Dropout, AlphaDropout

type Functional = Callable[[Tensor], Tensor]
type Tensors1T = tuple[Tensor]
type Tensors2T = tuple[Tensor, Tensor]
type Tensors3T = tuple[Tensor, Tensor, Tensor]
type Tensors4T = tuple[Tensor, Tensor, Tensor, Tensor]
type Tensors = tuple[Tensor, ...]
type Dtype = dtype
type Device = device
type Drop = Dropout | AlphaDropout

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
    'Drop'
]

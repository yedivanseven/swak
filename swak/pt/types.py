from collections.abc import Callable
from pandas import DataFrame
from torch import Tensor, dtype, device
from torch.nn import Module

type Functional = Callable[[Tensor, ...], Tensor]
type Tensors1T = tuple[Tensor]
type Tensors2T = tuple[Tensor, Tensor]
type Tensors3T = tuple[Tensor, Tensor, Tensor]
type Tensors4T = tuple[Tensor, Tensor, Tensor, Tensor]
type TensorsT = tuple[Tensor, ...]
type Dtype = dtype
type Device = device

__all__ = [
    'DataFrame',
    'Tensor',
    'Module',
    'Functional',
    'Tensors1T',
    'Tensors2T',
    'Tensors3T',
    'Tensors4T',
    'TensorsT',
    'Dtype',
    'Device'
]

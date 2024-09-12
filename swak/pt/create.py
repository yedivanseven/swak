from typing import Any, overload
import torch as pt
from ..magic import ArgRepr
from .types import Tensor, Module, Dtype, Device, DataFrame

__all__ = [
    'Create',
    'AsTensor',
    'from_dataframe',
    'To'
]


class Create(ArgRepr):
    """Partial of the top-level PyTorch function ``tensor``.

    Parameters
    ----------
    dtype: dtype, optional
        Torch dtype of the tensor to create. Defaults to ``None``
    device: device, optional
        Torch device to create the tensor on. Defaults to ``None``
    requires_grad: bool, optional
        If autograd should record operations on the returned tensor.
        Defaults to ``False``.
    pin_memory: bool, optional
        If set, returned tensor would be allocated in the pinned memory.
        Works only for CPU tensors. Defaults to ``False``.

    """

    def __init__(
            self,
            dtype: Dtype | None = None,
            device: Device | None = None,
            requires_grad: bool = False,
            pin_memory: bool = False
    ) -> None:
        super().__init__(dtype, device, requires_grad, pin_memory)
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad
        self.pin_memory = pin_memory

    def __call__(self, data: Any) -> Tensor:
        """Create a PyTorch tensor with the cached (keyword) arguments.

        Parameters
        ----------
        data
            Any type of data a PyTorch tensor can be created from.

        Returns
        -------
        Tensor
            A PyTorch tensor.

        """
        return pt.tensor(
            data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            pin_memory=self.pin_memory
        )


class AsTensor(ArgRepr):
    """Partial of the top-level PyTorch function ``as_tensor``.

    Parameters
    ----------
    dtype: dtype, optional
        Torch dtype of the tensor to create. Defaults to ``None``
    device: device, optional
        Torch device to create the tensor on. Defaults to ``None``

    """

    def __init__(
            self,
            dtype: Dtype | None = None,
            device: Device | None = None
    ) -> None:
        super().__init__(dtype, device)
        self.dtype = dtype
        self.device = device

    def __call__(self, data: Any) -> Tensor:
        """Create a PyTorch tensor with the cached (keyword) arguments.

        Parameters
        ----------
        data
            Any type of data a PyTorch tensor can be created from.

        Returns
        -------
        Tensor
            A PyTorch tensor sharing the memory with ``data``.

        """
        return pt.as_tensor(data, dtype=self.dtype, device=self.device)


def from_dataframe(df: DataFrame) -> Tensor:
    """Convert a pandas dataframe to a PyTorch tensor.

    This is simply wrapper around the top-level PyTorch function ``from_numpy``
    function called on the dataframe's ``values`` attribute.

    Parameters
    ----------
    df: DataFrame
        The pandas dataframe to convert.

    Returns
    -------
    Tensor
        A PyTorch tensor sharing the memory with the dataframe's data.

    """
    return pt.from_numpy(df.values)



class To(ArgRepr):
    """Move or change a tensor or module to a different device or dtype.

    Parameters
    ----------
    target: device or dtype
        The device or dtype to move the tensor or module to.


    """

    def __init__(self, target: Device | Dtype) -> None:
        super().__init__(target)
        self.target = target

    @overload
    def __call__(self, inp: Tensor) -> Tensor:
        ...

    @overload
    def __call__(self, inp: Module) -> Module:
        ...

    def __call__(self, inp):
        """Move or change a tensor or module to the specified target.

        Parameters
        ----------
        inp: Tensor or Module
            The input tensor or device to move or change to the `target`.

        Returns
        -------
        Tensor or Module
            The `inp` moved or changed to the cached `target`.

        """
        return inp.to(self.target)

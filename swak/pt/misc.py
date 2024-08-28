from typing import Any
from .types import Tensor, TensorsT, Module

__all__ = [
    'identity',
    'Identity'
]


def identity(tensor: Tensor, **_: Any) -> Tensor | TensorsT:
    """Simply pass through the argument and ignore keyword arguments.

    This is a placeholder for instances where a default function is required.

    Parameters
    ----------
    tensor: Tensor
        Any argument (typically a tensor) to be passed straight through.

    Returns
    -------
    Tensor
        The tensor passed in as argument.

    """
    return tensor


class Identity(Module):
    """PyTorch module that passes a tensor right through, doing nothing.

    This is a placeholder for instances where a default module is required.
    Providing any number of (keyword) arguments on instantiation is permitted,
    but they will be ignored.

    """

    def __init__(self, *_, **__) -> None:
        super().__init__()

    def forward(self, tensor: Tensor, **_: Any) -> Tensor | TensorsT:
        """Simply pass through the argument and ignore keyword arguments.

        Parameters
        ----------
        tensor: Tensor
            Any argument (typically a tensor) to be passed straight through.

        Returns
        -------
        Tensor
            The tensor passed in as argument.

        """
        return tensor

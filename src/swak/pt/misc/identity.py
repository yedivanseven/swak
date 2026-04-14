from typing import Any
from ..types import Tensor, Resettable


def identity(tensor: Tensor, *_: Any, **__: Any) -> Tensor:
    """Pass through the first argument, ignore additional (keyword) arguments.

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


class Identity(Resettable):
    """PyTorch module that passes a tensor right through, doing nothing.

    This is a placeholder for instances where a default ``Module`` is required
    that also has a :meth:`reset_parameters()` method. Providing any number
    of (keyword) arguments on instantiation is permitted, but they are ignored.

    """

    def __init__(self, *_: Any, **__: Any) -> None:
        super().__init__()

    def forward(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        """Pass through first argument, ignore additional (keyword) arguments.

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

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

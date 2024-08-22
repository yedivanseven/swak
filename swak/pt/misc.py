from typing import Any
from .types import Tensor, TensorsT, Module

__all__ = [
    'Identity'
]


class Identity(Module):
    """PyTorch module that passes tensors right through, doing nothing.

    To stay consistent with the PyTorch API, this is a class (instantiated
    with nothing) with callable instances rather than a function.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensors: Tensor, **_: Any) -> Tensor | TensorsT:
        """Simply pass through tensor argument(s) and ignore keyword arguments.

        Parameters
        ----------
        *tensors: Tensor
            Any number of tensors to be passed straight through.

        Returns
        -------
        Tensor or tuple
            A single tensor if only a single tensor was passed in, an empty
            tuple if none were passed in, and a tuple of tensors if more than
            one was passed in.

        """
        return tensors[0] if len(tensors) == 1 else tensors
from typing import Any, Self
from ..types import Block, Tensor


class IdentityBlock(Block):
    """PyTorch module that passes a tensor right through, doing nothing.

    This is a placeholder for instances where a default ``Module`` is required
    that not only has a :meth:`reset_parameters()` method, but also a
    :meth:`new()` method in addition to ``mod_dim``, ``device`` and ``dtype``.
    Providing any number of (keyword) arguments on instantiation is permitted,
    but they are ignored.

    Parameters
    ----------
    mod_dim: int
        Ignored but mandatory to maintain API compatibility.

    """

    def __init__(self, mod_dim: int, *_: Any, **__: Any) -> None:
        super().__init__()
        self.__mod_dim = mod_dim

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> None:
        """Just for API compatibility. Always returns None."""
        return None

    @property
    def dtype(self) -> None:
        """Just for API compatibility. Always returns None."""
        return None

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

    def new(self) -> Self:
        """Return a fresh, new instance.

        Providing any number of (keyword) arguments is permitted, but they will
        be ignored.

        Returns
        -------
        IdentityBlock
            A fresh, new instance of itself.

        """
        return self.__class__(self.mod_dim)

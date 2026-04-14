from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Block
from .skip import SkipConnection


class Repeat(Block):
    """Repeat a skip-connection, distilling ever finer detail from your data.

    Parameters
    ----------
    skip: SkipConnection
        An instance of a :class:`SkipConnection` to repeat.
    n_layers: int, optional
        How often to repeat the `skip`. Defaults to 2.
    device: str or torch.device, optional
        Torch device to first create the blocks on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the blocks in. Defaults to ``torch.float``.

    Raises
    ------
    TypeError
        If `n_layers` is not an integer.
    ValueError
        If `n_layers` is smaller than 1.

    Note
    ----
    If the skip-connection sets `norm_first` to ``True``, no norm will be
    applied to the final output of the last repetition. If a trailing norm
    is desired, it should be applied externally, after this module.

    See Also
    --------
    SkipConnection

    """

    def __init__(
            self,
            skip: SkipConnection,
            n_layers: int = 2,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.n_layers = self.__valid(n_layers)
        self.blocks = ptn.Sequential(*[
            skip.new().to(device, dtype)
            for _ in self.layers
        ])

    @staticmethod
    def __valid(n_layers: Any) -> int:
        """Make sure `n_layers` is a positive integer."""
        try:
            as_int = int(n_layers)
        except (ValueError, TypeError) as error:
            cls = type(n_layers).__name__
            tmp = '"{}" must at least be convertible to an int, unlike {}'
            msg = tmp.format('n_layers', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) 1, unlike {}!'
            msg = tmp.format('n_layers', as_int)
            raise ValueError(msg)
        return as_int

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.blocks[0].mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.blocks[0].device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.blocks[0].dtype

    @property
    def layers(self) -> range:
        """Range of layer indices."""
        return range(self.n_layers)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a stack of identical skip-connection blocks.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.blocks(inp)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of all blocks."""
        for block in self.blocks:
            block.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Repeat
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.blocks[0],
            self.n_layers,
            self.device,
            self.dtype
        )

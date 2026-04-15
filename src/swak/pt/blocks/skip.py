from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ..types import Module, Tensor, Block
from ..misc import Identity


class SkipConnection(Block):
    """Add a residual/skip connection around the wrapped neural-network block.

    Parameters
    ----------
    block: Block
        The block to wrap the residual/skip connection around. The reason why
        this cannot simply be a ``Module`` is that currently PyTorch does not
        provide a reasonable way of cloning them.
    dropout: float, optional
        The amount of dropout to apply to the block's output before adding it
        back to the activated residual. Defaults to 0.
    norm_first: bool, optional
        If ``True``, normalize the inputs before passing them through the block
        and adding the outputs to the raw inputs. If ``False``, pass inputs
        through the block first and normalize the sum of the inputs and outputs
        afterward. Defaults to ``True``.
    norm_cls: type, optional
        The class of the norm to be applied after adding input to output, e.g.,
        ``LayerNorm`` or ``BatchNorm1d``. Again, this is needed to easily
        create a fresh, new instances with equal, but independent parameters.
        Defaults to :class:`Identity`, resulting in no normalization.
    *args
        Arguments used to initialize an instance of `norm_cls`.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.
    **kwargs
        Keyword arguments used to initialize an instance of `norm_cls`.

    See Also
    --------
    ~swak.pt.misc.Identity

    """

    def __init__(
            self,
            block: Block,
            dropout: float = 0.0,
            norm_first: bool = True,
            norm_cls: type[Module] = Identity,
            *args: Any,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.block = block.to(device, dtype)
        self.dropout = dropout
        self.drop = ptn.Dropout(dropout)
        self.norm_first = norm_first
        self.norm_cls: type[Module] = norm_cls
        self.args = args
        self.kwargs = kwargs
        self.norm = norm_cls(*args, device=device, dtype=dtype, **kwargs)

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.block.mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.block.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.block.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a block with the input added to the output.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        if self.norm_first:
            return inp + self.drop(self.block(self.norm(inp)))
        return self.norm(inp + self.drop(self.block(inp)))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block and the norm."""
        self.block.reset_parameters()
        self.norm.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        SkipConnection
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.block.new(),
            self.dropout,
            self.norm_first,
            self.norm_cls,
            *self.args,
            device=self.device,
            dtype=self.dtype,
            **self.kwargs
        )

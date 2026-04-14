from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Module, Tensor, Functional, Block


class GatedHiddenBlock(Block):
    """A configurable, gated linear unit (GLU) with single hidden layer.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    gate: Module or function, optional
        The activation function to be applied to half of the (linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    factor: int, optional
        The size of the hidden layer is this integer factor times `mod_dim`.
        Defaults to 4.
    bias: bool, optional
        Whether to add a learnable bias vector in to the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            factor: int = 4,
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.gate = self._reset(gate, device, dtype)
        self.factor = factor
        self.bias = bias
        self.widen = ptn.Linear(
            in_features=mod_dim,
            out_features=2 * self.dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.shrink = ptn.Linear(
            in_features=self.dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.widen.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.widen.weight.dtype

    @property
    def dim(self) -> int:
        """The hidden dimension after gating."""
        return (self.factor * self.mod_dim) // 2

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a gated linear unit (GLU) with a hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        widened = self.widen(inp)
        gated = widened[..., :self.dim] * self.gate(widened[..., self.dim:])
        return self.shrink(gated)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.gate = self._reset(self.gate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        GatedHiddenBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.gate,
            self.factor,
            self.bias,
            self.device,
            self.dtype
        )

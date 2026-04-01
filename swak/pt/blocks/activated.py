from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Module, Tensor, Functional, Block


class ActivatedBlock(Block):
    """A single, non-linearly activated layer.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    activate: Module or function, optional
        The activation function to be applied after projecting into higher-
        dimensional space. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        ``torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to ``ELU()``.
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
            activate: Module | Functional = ptn.ELU(),
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(activate, device, dtype)
        self.bias = bias
        self.project = ptn.Linear(
            in_features=mod_dim,
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
        return self.project.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.project.weight.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single, non-linearly activated hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.activate(self.project(inp))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.project.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(self.activate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        ActivatedBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.bias,
            self.device,
            self.dtype
        )

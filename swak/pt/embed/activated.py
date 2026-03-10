from typing import Self
import torch as pt
import torch.nn as ptn
from ..misc import identity
from ..types import Tensor, Module, Functional, Block


class ActivatedEmbedder(Block):
    """Simple linear projection of an individual feature into embedding space.

    Parameters
    ----------
    mod_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    activate: Module or function, optional
        The activation function to be applied after (linear) projection into
        embedding space. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        `torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to ``identity``, resulting in no
        non-linear activation whatsoever.
    bias: bool, optional
        Whether to add a learnable bias vector in the projection.
        Defaults to ``True``.
    inp_dim: int, optional
        The number of features to embed together. Defaults to 1.
    device: str or pt.device, optional
        Torch device to first create the embedder on. Defaults to "cpu".
    dtype: pt.dtype, optional
        Torch dtype to first create the embedder in.
        Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            activate: Module | Functional = identity,
            bias: bool = True,
            inp_dim: int = 1,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
            self.activate = activate.to(device=device, dtype=dtype)
        else:
            self.activate = activate
        self.bias = bias
        self.inp_dim = inp_dim
        self.embed = ptn.Linear(inp_dim, mod_dim, bias, device, dtype)

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.embed.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.embed.weight.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Embed a single numerical feature through a (non-)linear projection.

        Parameters
        ----------
        inp: Tensor
            The last dimension of the input tensor is typically expected to be
            of size 1 and to contain the numerical value of a single feature.
            In case `inp_dim` dim was explicitly set to a value > 1 on
            instantiation, the size of the last dimension must match `inp_dim`,
            the number of numerical features to embed together.

        Returns
        -------
        Tensor
            The output has the same number of dimensions as the input with
            the size of the last dimension changed to the specified `mod_dim`.

        """
        return self.activate(self.embed(inp))

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.embed.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        ActivatedEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.bias,
            self.inp_dim,
            self.device,
            self.dtype
        )

from typing import Any, Self
import torch.nn as ptn
from ..types import Tensor, Module, Resettable, Functional


class GatedEmbedder(Resettable):
    """Flexible Gated Linear Unit (GLU) for embedding a numerical feature.

    Parameters
    ----------
    mod_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    gate: Module or function, optional
        The activation function to be applied to half of the (linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    inp_dim: int, optional
        The number of features to embed together. Defaults to 1.
    **kwargs
        Additional keyword arguments to pass through to the linear layer.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            inp_dim: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.gate = gate
        self.inp_dim = inp_dim
        self.kwargs = kwargs
        self.embed = ptn.Linear(inp_dim, 2 * mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Embed a single numerical feature through a Gated Linear Unit (GLU).

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
        emb = self.embed(inp)
        return emb[..., :self.mod_dim] * self.gate(emb[..., self.mod_dim:])

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.embed.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            gate: Module | Functional | None = None,
            inp_dim: int | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Desired embedding size. Will become the size of the last dimension
            of the output tensor. Overwrites the `mod_dim` of the current
            instance if given. Defaults to ``None``.
        gate: Module or function, optional
            The activation function to be applied to half of the (linearly)
            projected input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `gate` of the current instance if given.
            Defaults to ``None``.
        inp_dim: int, optional
            The number of features to embed together. Overwrites the `inp_dim`
            of the current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        GatedEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.gate if gate is None else gate,
            self.inp_dim if inp_dim is None else inp_dim,
            **(self.kwargs | kwargs)
        )

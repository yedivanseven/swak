from typing import Any, Self
import torch.nn as ptn
from ..types import Tensor, Module


class LinearEmbedder(Module):
    """Simple linear projection of an individual feature into embedding space.

    Parameters
    ----------
    mod_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    inp_dim: int, optional
        The number of features to embed. Defaults to 1.
    **kwargs
        Additional keyword arguments to pass through to the linear layer.

    """

    def __init__(
            self,
            mod_dim: int,
            inp_dim: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.inp_dim = inp_dim
        self.kwargs = kwargs
        self.embed = ptn.Linear(inp_dim, mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Embed a single numerical feature through a linear projection.

        Parameters
        ----------
        inp: Tensor
            Input tensor with at least 2 dimensions. The last dimension is
            typically expected to be of size 1 and to contain the numerical
            value of a single feature. In case `inp_dim` dim was explicitly
            set to a value > 1 on instantiation, the size of the last
            dimension must match `inp_dim`, the number of numerical features
            to embed together.

        Returns
        -------
        Tensor
            The output has the same number of dimensions as the input with the
            size of the last dimension changed to the specified `out_dim`.

        """
        return self.embed(inp)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.embed.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            inp_dim: int | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Desired embedding size. Will become the size of the last dimension
            of the output tensor. Overwrites the `out_dim` of the current
            instance if given. Defaults to ``None``.
        inp_dim: int, optional
            The number of features to embed. Overwrites the `inp_dim` of the
            current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        LinearEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.inp_dim if inp_dim is None else inp_dim,
            **(self.kwargs | kwargs)
        )

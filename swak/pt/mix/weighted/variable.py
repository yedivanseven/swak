from typing import Self
import torch as pt
import torch.nn as ptn
from ...types import Tensor, Resettable


class VariableSumMixer(Resettable):
    """Combine stacked feature vectors through a learnable linear combination.

    Specifically, a single, global set of linear-combination coefficients is
    learned. These coefficients sum to 1 and can thus be seen as some sort of
    feature importance.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.

    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.coeffs = ptn.Parameter(pt.ones(n_features))
        self.norm = ptn.Softmax(dim=-1)

    def importance(self, inp: Tensor) -> Tensor:
        """Learned, global feature weights in the normed sum over all features.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension is
            expected to contain the features vectors themselves.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input with the
            last dimension being dropped.

        """
        return self.norm(self.coeffs.expand(*inp.shape[:-1]))

    def forward(self, inp: Tensor) -> Tensor:
        """Linearly combine stacked feature vectors with global coefficients.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension is
            expected to contain the features vectors themselves.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input.
            The next-to-last dimension is dropped and the last dimension now
            contains the (normed) linear combination of all feature vectors.

        """
        return (self.importance(inp).unsqueeze(dim=-2) @ inp).squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize the coefficients for the linear combination."""
        self.coeffs.data = pt.ones(self.n_features)

    def new(self, n_features: int | None = None) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        n_features: int, optional
            The number of features to combine. Must be equal to the size of the
            next-to-last dimension of the input tensor. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.

        Returns
        -------
        VariableSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )

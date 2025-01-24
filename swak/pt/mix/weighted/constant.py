from typing import Self
import torch as pt
from ...types import Tensor, Resettable


class ConstantSumMixer(Resettable):
    """Combine stacked feature vectors by simply adding them.

    The sum is then "normed" through dividing by the number of features.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.

    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.register_buffer('coeffs', pt.tensor(1.0 / n_features))

    def importance(self, inp: Tensor) -> Tensor:
        """Constant feature weights in the normed sum over all features.

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
        return self.coeffs.expand(*inp.shape[:-1])

    def forward(self, inp: Tensor) -> Tensor:
        """Add stacked feature vectors with constant and equal weights.

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
            contains the (normed) sum of all feature vectors.

        """
        return (self.importance(inp).unsqueeze(dim=-2) @ inp).squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

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
        ConstantSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )

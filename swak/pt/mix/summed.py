from typing import Self
import torch as pt
from ..types import Tensor, Module


class ArgsSumMixer(Module):
    """Combined multiple feature tensors by simply summing them up.

    The sum is then normalized by the square root of the number of tensors.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal the number of
        arguments instances are called with.

    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self._rsqrt = pt.tensor(n_features).rsqrt()

    def forward(self, *inps: Tensor) -> Tensor:
        """Forward pass for combining multiple feature tensors into one.

        Parameters
        ----------
        *inps: Tensor
            Each input tensor represents one feature with the size of the last
            dimension representing the length of the feature vector. The size
            of this and all other dimensions must be the same for all `inps`.
            The number of call arguments must match the `n_features` specified
            at instantiation.

        Returns
        -------
        Tensor
            The output tensor has the same dimensions as any input tensor.

        """
        return pt.stack(inps, dim=-2).sum(dim=-2) * self._rsqrt

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

    def new(self, n_features: int | None = None) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        n_features: int, optional
            The number of features to combine. Must be equal the number of
            arguments instances are called with. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.

        Returns
        -------
        ArgsSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )


class StackSumMixer(Module):
    """Combined stacked feature vectors by simply summing them up.

    The sum is then normalized by the square root of the number of features.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.

    """

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self._rsqrt = pt.tensor(n_features).rsqrt()

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            The size of the next-to-last last dimension of the input tensor is
            expected to match the `n_features` provided at instantiation.
            The last dimension is expected to contain the features vectors.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input.
            The next-to-last dimension is dropped and the last dimension
            now carries the normed sum of all features.

        """
        return inp.sum(dim=-2) * self._rsqrt

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
        StackSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )

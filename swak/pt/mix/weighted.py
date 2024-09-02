from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module


class ArgsWeightedSumMixer(Module):
    """Combine multiple feature tensors through a (normed) linear combination.

    The coefficients of the linear combinations are learned during training
    and sum to 1. They can, thus, be seen as a sort of feature importance.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal to the number of
        arguments instances are called with.

    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self._coefficients = ptn.Parameter(pt.ones(n_features))
        self.norm = ptn.Softmax(dim=0)

    @property
    def importance(self) -> Tensor:
        """Normalized coefficients of the linear combination of features."""
        return self.norm(self._coefficients).detach()

    def forward(self, *inps: Tensor) -> Tensor:
        """Learn coefficients for linearly combining input feature tensors.

        Parameters
        ----------
        *inps: tensor
            Each input tensor represents one feature with the size of the last
            dimension representing the length of the feature vector. The size
            of this and all other dimensions must be the same for all `inps`.
            The number of call arguments must match the `n_features` specified
            at instantiation.

        Returns
        -------
        Tensor:
            The output tensor has the same dimensions as any input tensor.

        """
        return self.norm(self._coefficients) @  pt.stack(inps, dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize the coefficients of the linear combination."""
        self._coefficients = ptn.Parameter(pt.ones(self.n_features))

    def new(self, n_features: int | None = None) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        n_features: int, optional
            The number of features to combine. Must be equal to the number of
            arguments instances are called with. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.

        Returns
        -------
        ArgsWeightedSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )


class StackWeightedSumMixer(Module):
    """Combine stacked feature vectors through a (normed) linear combination.

    The coefficients of the linear combinations are learned during training
    and sum to 1. They can, thus, be seen as a sort of feature importance.

    Parameters
    ----------
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.

    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features
        self._coefficients = ptn.Parameter(pt.ones(n_features))
        self.norm = ptn.Softmax(dim=0)

    @property
    def importance(self) -> Tensor:
        """Normalized coefficients of the linear combination of features."""
        return self.norm(self._coefficients).detach()

    def forward(self, inp: Tensor) -> Tensor:
        """Learn coefficients for linearly combining stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            The size of the next-to-last last dimension of the input tensor is
            expected to match the `n_features` provided at instantiation.
            The last dimension is expected to contain the features vectors.

        Returns
        -------
        Tensor:
            The output tensor has one fewer dimensions than the input.
            The next-to-last dimension is dropped and the size of the last
            dimension is once again the size of feature space.

        """
        return self.norm(self._coefficients) @  inp

    def reset_parameters(self) -> None:
        """Re-initialize the coefficients of the linear combination."""
        self._coefficients = ptn.Parameter(pt.ones(self.n_features))

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
        StackWeightedSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.n_features if n_features is None else n_features
        )

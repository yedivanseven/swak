from typing import Any, Self
import torch.nn as ptn
from ...types import Tensor, Module, Functional, Resettable
from ...misc import identity


class ActivatedSumMixer(Resettable):
    """Combine stacked feature vectors by a per-instance linear combination.

    The per-instance coefficients sum to 1 for each data point and can thus be
    seen as some sort of per-instance feature importance. They are obtained by
    concatenating all features into a single, wide vector, linearly projecting
    down to a vector with the same number of elements as there are features to
    combine, optionally activating, and then applying a softmax.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.
    activate: Module or function, optional
        The activation function to be applied after (linearly) mixing the
        concatenated features. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        `torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to ``identity``, resulting in no
        non-linear activation whatsoever.
    **kwargs
        Keyword arguments are passed on to the linear layer.

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            activate: Module | Functional = identity,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.activate = activate
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        self.kwargs = kwargs
        self.coeffs = ptn.Linear(n_features * mod_dim, n_features, **kwargs)
        self.norm = ptn.Softmax(dim=-1)

    def importance(self, inp: Tensor) -> Tensor:
        """Per-instance weights in the normed linear combination of features.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input with the
            last dimension being dropped.

        """
        return self.norm(self.activate(self.coeffs(inp.flatten(start_dim=-2))))

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            Feature vectors stacked into a tensor of at least 2 dimensions.
            The size of the next-to-last last dimension is expected to match
            the `n_features` provided at instantiation. The last dimension
            (of size `mod_dim`) is expected to contain the features vectors.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input.
            The next-to-last dimension is dropped and the last dimension now
            contains the per-instance (normed) linear combination of all
            feature vectors.

        """
        return (self.importance(inp).unsqueeze(dim=-2) @ inp).squeeze(dim=-2)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.coeffs.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            n_features: int | None = None,
            activate: Module | Functional | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Size of the feature space. The input tensor is expected to be of
            that size in its last dimension and the output will again have this
            size in its last dimension. Overwrites the `mod_dim` of the current
            instance if given. Defaults to ``None``.
        n_features: int, optional
            The number of features to combine. Must be equal to the size of the
            next-to-last dimension of the input tensor. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.
        activate: Module or function, optional
            The activation function to be applied after (linearly) mixing the
            concatenated features. Must be a callable that accepts a tensor as
            sole argument, like a module from ``torch.nn`` or a function from
            `torch.nn.functional``, depending on whether it needs to be further
            parameterized or not. Overwrites the `activate` of the current
            instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        ActivatedSumMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.n_features if n_features is None else n_features,
            self.activate if activate is None else activate,
            **(self.kwargs | kwargs)
        )

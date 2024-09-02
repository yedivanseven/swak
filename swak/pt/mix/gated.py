from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module, Functional


class GatedArgsConcatMixer(Module):
    """Combined multiple feature tensors through a gated linear unit (GLU).

    Multiple tensors are concatenated to produce a single, wide feature vector,
    which is then projected into a space twice the size of the model. One half
    is passed through an (optional) activation function to gate the other half,
    thus reducing the final output back down to the model size.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. Each feature vector is expected
        to be of that size in its last dimension at the input and the output
        will again have this size in its last dimension.
    n_features: int
        The number of features to combine. Must be equal to the number of
        arguments instances are called with.
    gate: Module or function, optional
        The activation function to be applied to half of the (linearly)
        transformed inputs before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``.
        Defaults to a sigmoid.
    **kwargs
        Keyword arguments are passed on to the linear layer.

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            gate: Module | Functional = ptn.Sigmoid(),
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.gate = gate
        self.kwargs = kwargs
        self.mix = ptn.Linear(n_features * mod_dim, 2 * mod_dim, **kwargs)

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
        mixed = self.mix(pt.cat(inps, dim=-1))
        return mixed[..., :self.mod_dim] * self.gate(mixed[..., self.mod_dim:])

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.mix.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            n_features: int | None = None,
            gate: Module | Functional | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Size of the feature space. Each feature vector is expected
            to be of that size in its last dimension. Overwrites the `mod_dim`
            of the current instance if given. Defaults to ``None``.
        n_features: int, optional
            The number of features to combine. Must be equal to the number of
            arguments instances are called with. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.
        gate: Module or function, optional
            The activation function to be applied to half of the (linearly)
            transformed input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `gate` of the current instance if given.
            Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        GatedArgsConcatMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.n_features if n_features is None else n_features,
            self.gate if gate is None else gate,
            **(self.kwargs | kwargs)
        )


class GatedStackConcatMixer(Module):
    """Combined stacked feature vectors through a gated linear unit (GLU).

    Features are concatenated into a single, wide vector and projected into a
    space twice the size of the model. One half is then passed through an
    (optional) activation function to gate the other half, thus reducing
    the final output back down to the model size.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. Each feature vector is expected
        to be of that size in its last dimension at the input and the output
        will again have this size in its last dimension.
    n_features: int
        The number of features to combine. Must be equal to the size of the
        next-to-last dimension of the input tensor.
    gate: Module or function, optional
        The activation function to be applied to half of the (linearly)
        transformed inputs before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``.
        Defaults to a sigmoid.
    **kwargs
        Keyword arguments are passed on to the linear layer.

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            gate: Module | Functional = ptn.Sigmoid(),
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        self.gate = gate
        self.kwargs = kwargs
        self.mix = ptn.Linear(n_features * mod_dim, 2 * mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass for combining multiple stacked feature vectors.

        Parameters
        ----------
        inp: Tensor
            The size of the next-to-last last dimension of the input tensor is
            expected to match the `n_features` provided at instantiation.
            The last dimension (of size `mod_dim`) is expected to contain the
            features vectors themselves.

        Returns
        -------
        Tensor
            The output tensor has one fewer dimensions than the input.
            The next-to-last dimension is dropped and the size of the last
            dimension is once again `mod_dim`.

        """
        mixed = self.mix(inp.flatten(start_dim=-2))
        return mixed[..., :self.mod_dim] * self.gate(mixed[..., self.mod_dim:])

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.mix.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            n_features: int | None = None,
            gate: Module | Functional | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            Size of the feature space. Each feature vector is expected
            to be of that size in its last dimension. Overwrites the `mod_dim`
            of the current instance if given. Defaults to ``None``.
        n_features: int, optional
            The number of features to combine. Must be equal to the size of the
            next-to-last dimension of the input tensor. Overwrites `n_features`
            of the current instance if given. Defaults to ``None``.
        gate: Module or function, optional
            The activation function to be applied to half of the (linearly)
            transformed input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `gate` of the current instance if given.
            Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layer together.

        Returns
        -------
        GatedStackConcatMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.n_features if n_features is None else n_features,
            self.gate if gate is None else gate,
            **(self.kwargs | kwargs)
        )

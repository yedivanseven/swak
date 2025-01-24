from typing import Any, Self
import torch.nn as ptn
from ..types import Tensor, Module, Functional, Drop, Resettable


class GatedResidualConcatMixer(Resettable):
    """Combined stacked feature vectors through a Gated Residual Network (GRN).

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
        The activation function to be applied after (linear) transformation,
        but prior to gating. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        `torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to an ``ELU`` activation.
    gate: Module or function, optional
        The activation function to be applied to half of the (non-linearly)
        transformed input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    drop: Module, optional
        Typically an instance of ``Dropout`` or ``AlphaDropout``. Defaults to
        ``Dropout(p=0.0)``, resulting in no dropout being applied.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    Note
    ----
    This implementation is inspired by how features are encoded in `Temporal
    Fusion Transformers`, [1]_ but it is not quite the same. Specifically, the
    intermediate linear layer (Eq. 3) is eliminated and dropout is applied
    directly to the activations after the first layer. Also, the layer norm
    (Eq. 2) is replaced by simply dividing the sum of (linearly projected)
    input and gated signal by 2. Should additional normalization be desired,
    it can be performed independently on the output of this module.

    References
    ----------
    .. [1] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, `Temporal Fusion
           Transformers for Interpretable Multi-horizon Time Series
           Forecasting`, `arXiv:1912.09363v3 <https://arxiv.org/abs/
           1912.09363>`__ (2020).

    """

    def __init__(
            self,
            mod_dim: int,
            n_features: int,
            activate: Module | Functional = ptn.ELU(),
            gate: Module | Functional = ptn.Sigmoid(),
            drop: Drop = ptn.Dropout(0.0),
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_features = n_features
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.kwargs = kwargs
        self.project = ptn.Linear(n_features * mod_dim, mod_dim, **kwargs)
        self.widen = ptn.Linear(mod_dim, 2 * mod_dim, **kwargs)

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
        projected = self.project(inp.flatten(start_dim=-2))
        wide = self.widen(self.drop(self.activate(projected)))
        gated = wide[..., :self.mod_dim] * self.gate(wide[..., self.mod_dim:])
        return 0.5 * (projected + gated)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.project.reset_parameters()
        self.widen.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            n_features: int | None = None,
            activate: Module | Functional | None = None,
            gate: Module | Functional | None = None,
            drop: Drop | None = None,
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
            The activation function to be applied after (linear) transform,
            but prior to gating. Must be a callable that accepts a tensor as
            sole argument, like a module from ``torch.nn`` or a function from
            `torch.nn.functional``, depending on whether it needs to be further
            parameterized or not. Overwrites `activate` of the current instance
            if given. Defaults to ``None``.
        gate: Module or function, optional
            The activation function to be applied to half of the (linearly)
            transformed input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `gate` of the current instance if given.
            Defaults to ``None``.
        drop: Module, optional
            Typically an instance of ``Dropout`` or ``AlphaDropout``.
            Overwrites the `drop` of the current instance if given.
            Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layers together.

        Returns
        -------
        GatedResidualConcatMixer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.n_features if n_features is None else n_features,
            self.activate if activate is None else activate,
            self.gate if gate is None else gate,
            self.drop if drop is None else drop,
            **(self.kwargs | kwargs)
        )

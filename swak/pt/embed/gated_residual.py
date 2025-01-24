from typing import Any, Self
import torch.nn as ptn
from ..types import Tensor, Module, Functional, Drop, Resettable


class GatedResidualEmbedder(Resettable):
    """Gated Residual Network (GRN) for embedding numerical features.

    Parameters
    ----------
    mod_dim: int
        Desired embedding size. Will become the size of the last dimension of
        the output tensor.
    activate: Module or function, optional
        The activation function to be applied after (linear) projection into
        embedding space, but prior to gating. Must be a callable that accepts
        a tensor as sole argument, like a module from ``torch.nn`` or a
        function from `torch.nn.functional``, depending on whether it needs to
        be further parameterized or not. Defaults to an ``ELU`` activation.
    gate: Module or function, optional
        The activation function to be applied to half of the (non-linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    drop: Module, optional
        Typically an instance of ``Dropout`` or ``AlphaDropout``. Defaults to
        ``Dropout(p=0.0)``, resulting in no dropout being applied.
    inp_dim: int, optional
        The number of features to embed together. Defaults to 1.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    Note
    ----
    This implementation is inspired by how features are encoded in `Temporal
    Fusion Transformers`, [1]_ but it is not quite the same. Firstly, the
    (linear) projection of scalar numerical features into embedding space
    happens inside the present module. Secondly, this embedding vector is not
    transformed again (as Eq. 4 seems to imply) and there is no option to add
    a `context vector`. Thirdly, the intermediate linear layer (Eq. 3) is
    eliminated and dropout is applied directly to the activations after the
    first layer. Finally, the layer norm (Eq. 2) is replaced by simply
    dividing the sum of (linearly projected) input and gated signal by 2.
    Should additional normalization be desired, it can be performed
    independently on the output of this module.

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
            activate: Module | Functional = ptn.ELU(),
            gate: Module | Functional = ptn.Sigmoid(),
            drop: Drop = ptn.Dropout(0.0),
            inp_dim: int = 1,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.inp_dim = inp_dim
        self.kwargs = kwargs
        self.project = ptn.Linear(inp_dim, mod_dim, **kwargs)
        self.widen =  ptn.Linear(mod_dim, 2 * mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Embed a numerical feature through a Gated Residual Network (GRN).

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
        projected = self.project(inp)
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
            activate: Module | Functional | None = None,
            gate: Module | Functional | None = None,
            drop: Drop | None = None,
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
        activate: Module or function, optional
            The activation function to be applied after (linear) projection
            into embedding space, but prior to gating. Must be a callable that
            accepts a tensor as sole argument, like a module from ``torch.nn``
            or a function from `torch.nn.functional``, depending on whether it
            needs to be further parameterized or not. Overwrites the `activate`
            of the current instance if given. Defaults to ``None``.
        gate: Module or function, optional
            The activation function to be applied to half of the (non-linearly)
            projected input before multiplying with the other half. Must be
            a callable that accepts a tensor as sole argument, like a module
            from ``torch.nn`` or a function from ``torch.nn.functional``.
            Overwrites the `gate` of the current instance if given.
            Defaults to ``None``.
        drop: Module, optional
            Typically an instance of ``Dropout`` or ``AlphaDropout``.
            Overwrites the `drop` of the current instance if given.
            Defaults to ``None``.
        inp_dim: int, optional
            The number of features to embed together. Overwrites the `inp_dim`
            of the current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layers together.

        Returns
        -------
        GatedResidualEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.activate if activate is None else activate,
            self.gate if gate is None else gate,
            self.drop if drop is None else drop,
            self.inp_dim if inp_dim is None else inp_dim,
            **(self.kwargs | kwargs)
        )

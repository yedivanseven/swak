from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module, Functional, Drop


class GatedResidualEmbedder(Module):
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

    Notes
    -----
    This implementation is inspired by how features are encoded in `Temporal
    Fusion Transformers`, [1]_ but it is not quite the same. Firstly, the
    (linear) projection of features into embedding space happens inside the
    present module with no option to add a `context vector`. Secondly, the
    intermediate linear layer (Eq. 3) is eliminated and dropout is applied
    directly to the activations after the first layer. Finally, the layer
    norm (Eq. 2) is replaced by simply dividing the sum of (linearly projected)
    input and gated signal by the square root of 2.


    References
    ----------
    .. [1] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, `Temporal Fusion
           Transformers for Interpretable Multi-horizon Time Series
           Forecasting`, `arXiv:1912.09363v3 <https://arxiv.org/abs/
           1912.09363>`__ (2020).

    See Also
    --------
    swak.pt.misc.Identity

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
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.inp_dim = inp_dim
        self.kwargs = kwargs
        self.project = ptn.Linear(inp_dim, mod_dim, **kwargs)
        self.expand =  ptn.Linear(mod_dim, 2 * mod_dim, **kwargs)
        self._rsqrt2 = pt.tensor(2).rsqrt()

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
            The output has the same number of dimensions as the input with the
            size of the last dimension changed to the specified `out_dim`.

        """
        projected = self.project(inp)
        exp = self.expand(self.drop(self.activate(projected)))
        gated = exp[..., :self.mod_dim] * self.gate(exp[..., self.mod_dim:])
        return (projected + gated) * self._rsqrt2

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.project.reset_parameters()
        self.expand.reset_parameters()

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
            of the output tensor. Overwrites the `out_dim` of the current
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
            layer together.

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

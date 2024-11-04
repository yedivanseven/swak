"""Flexible and composable building blocks for constructing neural-networks.

After features are embedded and combined, it is time to extract as much
information as possible to predict the desired target. One way of doing this
systematically is to repeat layers of identical internal architecture with
residual (or skip) connections between them.

"""

from typing import Any, Self
from abc import ABC, abstractmethod
import torch.nn as ptn
from .types import Module, Tensor, Functional, Drop
from .misc import Identity

__all__ = [
    'Block',
    'ActivatedBlock',
    'GatedBlock',
    'GatedResidualBlock',
    'SkipConnection',
    'Repeat'
]


class Block(Module, ABC):
    """Abstract base class for stackable/repeatable neural-network components.

    The input and output tensors of such components must have the same
    dimensions and sizes!

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        ...

    @abstractmethod
    def reset_parameters(self) -> None:
        """Subclasses implement in-place reset of all internal parameters."""
        ...


class ActivatedBlock(Block):
    """A single, non-linearly activated hidden layer of configurable size.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    activate: Module or function, optional
        The activation function to be applied after projecting into higher-
        dimensional space. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        ``torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to ``ELU()``.
    hidden_factor: int, optional
        The size of the hidden layer is this integer factor times `mod_dim`.
        Defaults to 4.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    """

    def __init__(
            self,
            mod_dim: int,
            activate: Module | Functional = ptn.ELU(),
            hidden_factor: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.activate = activate
        self.hidden_factor = hidden_factor
        self.kwargs = kwargs
        self.widen = ptn.Linear(mod_dim, hidden_factor * mod_dim, **kwargs)
        self.shrink = ptn.Linear(hidden_factor * mod_dim, mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single, non-linearly activated hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.shrink(self.activate(self.widen(inp)))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the linear projections."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.hidden_factor,
            **self.kwargs
        )


class GatedBlock(Block):
    """A configurable, gated linear unit (GLU).

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    gate: Module or function, optional
        The activation function to be applied to half of the (linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.gate = gate
        self.kwargs = kwargs
        self.widen = ptn.Linear(mod_dim, 2 * mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single gated linear unit (GLU).

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        wide = self.widen(inp)
        return wide[..., :self.mod_dim] * self.gate(wide[..., self.mod_dim:])

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the linear projection."""
        self.widen.reset_parameters()


    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.gate,
            **self.kwargs
        )


class GatedResidualBlock(Block):
    """Gated Residual Network (GRN) for efficiently extracting information.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    activate: Module or function, optional
        The activation function to be applied after (linear) projection, but
        prior to gating. Must be a callable that accepts a tensor as sole
        argument, like a module from ``torch.nn`` or a function from
        `torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to an ``ELU`` activation.
    gate: Module or function, optional
        The activation function to be applied to half of the (non-linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    drop: Module, optional
        Dropout to be applied within the GRN. Typically an instance of
        ``Dropout`` or ``AlphaDropout``. Defaults to ``Dropout(p=0.0)``,
        resulting in no dropout being applied.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    Note
    ----
    This implementation is inspired by how features are encoded in `Temporal
    Fusion Transformers`, [1]_ but it is not quite the same. Firstly, the
    intermediate linear layer (Eq. 3) is eliminated and dropout is applied
    directly to the activations after the first layer. Secondly, the layer norm
    (Eq. 2) is replaced by simply dividing the sum of (linearly projected)
    input and gated signal by 2. Should additional normalization be desired, it
    can be performed independently on the output of this module.

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
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.kwargs = kwargs
        self.project = ptn.Linear(mod_dim, mod_dim, **kwargs)
        self.widen = ptn.Linear(mod_dim, 2 * mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single gated residual network (GRN).

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        projected = self.project(inp)
        wide = self.widen(self.drop(self.activate(projected)))
        gated = wide[..., :self.mod_dim] * self.gate(wide[..., self.mod_dim:])
        return 0.5 * (projected + gated)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the linear projections."""
        self.project.reset_parameters()
        self.widen.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.gate,
            self.drop,
            **self.kwargs
        )


class SkipConnection(Block):
    """Add a residual/skip connection around the wrapped neural-network block.

    Parameters
    ----------
    block: Block
        The block to wrap the residual/skip connection around. The reason why
        this cannot simply be a ``Module`` is that currently PyTorch does not
        provide a reasonable way of cloning them.
    drop: Module, optional
        Dropout to be applied to the output of `block` before adding it to
        its input. Typically an instance of ``Dropout`` or ``AlphaDropout``.
        Defaults to ``Dropout(p=0.0)``, resulting in no dropout being applied.
    norm_cls: type, optional
        The class of the norm to be applied after adding input to output, e.g.,
        ``LayerNorm`` or ``BatchNorm1d``. Again, this is needed to easily
        create a fresh, new instances with equal, but independent parameters.
        Defaults to ``Identity``, resulting in no normalization whatsoever.
    *args
        Arguments used to initialize an instance of `norm_cls`.
    **kwargs
        Keyword arguments used to initialize an instance of `norm_cls`.

    """

    def __init__(
            self,
            block: Block,
            drop: Drop = ptn.Dropout(0.0),
            norm_cls: type[Module] = Identity,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.block = block
        self.drop = drop
        self.norm_cls: type[Module] = norm_cls
        self.args = args
        self.kwargs = kwargs
        self.norm = norm_cls(*args, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a block with the input added to the output.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.norm(0.5 * (inp + self.drop(self.block(inp))))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block and the norm."""
        self.block.reset_parameters()
        self.norm.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.block.new(),
            self.drop,
            self.norm_cls,
            *self.args,
            **self.kwargs
        )


class Repeat(Block):
    """Repeat a skip-connection, distilling ever finer detail from your data.

    Parameters
    ----------
    skip: SkipConnection
        An instance of a ``SkipConnection`` to repeat.
    n_layers: int, optional
        How often to repeat the `skip`. Defaults to 2.

    """

    def __init__(self, skip: SkipConnection, n_layers: int = 2) -> None:
        super().__init__()
        self.skip = skip
        self.n_layers = n_layers
        self.sequence = ptn.Sequential(*[skip.new() for _ in self.layers])

    @property
    def layers(self) -> range:
        """Range of layer indices."""
        return range(self.n_layers)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a stack of identical skip-connection blocks.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.sequence(inp)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of all blocks."""
        for block in self.sequence:
            block.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.skip,
            self.n_layers
        )

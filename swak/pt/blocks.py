"""Flexible and composable building blocks for constructing neural-networks.

After features are embedded and combined, it is time to extract as much
information as possible to predict the desired target. One way of doing this
systematically is to repeat layers of identical internal architecture with
residual (or skip) connections between them.

"""

from typing import Any, Self
import torch.nn as ptn
from .types import Module, Tensor, Functional, Drop, Block
from .misc import Identity

__all__ = [
    'ActivatedBlock',
    'ActivatedHiddenBlock',
    'GatedBlock',
    'GatedHiddenBlock',
    'ActivatedGatedBlock',
    'GatedResidualBlock',
    'SkipConnection',
    'Repeat'
]


# ToDo: Rethink resetting parameters in "new"
class ActivatedBlock(Block):
    """A single, non-linearly activated layer.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension and the output will again have this size in
        its last dimension.
    activate: Module or function, optional
        The activation function to be applied after the affine transformation.
        Must be a callable that accepts a tensor as sole argument, like a
        module from ``torch.nn`` or a function from ``torch.nn.functional``,
        depending on whether it needs to be further parameterized or not.
        Defaults to ``ELU()``.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    """

    def __init__(
            self,
            mod_dim: int,
            activate: Module | Functional = ptn.ELU(),
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        self.activate = activate
        self.kwargs = kwargs
        self.project = ptn.Linear(self.mod_dim, self.mod_dim, **kwargs)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single, non-linearly activated layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.activate(self.project(inp))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the linear projections."""
        self.project.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.activate,
            **self.kwargs
        )


class ActivatedHiddenBlock(Block):
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
    drop: Module, optional
        Dropout to be applied after activation. Typically an instance of
        ``Dropout`` or ``AlphaDropout``. Defaults to ``Dropout(p=0.0)``,
        resulting in no dropout being applied.
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
            drop: Drop = ptn.Dropout(0.0),
            hidden_factor: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        self.activate = activate
        self.drop = drop
        self.hidden_factor = round(hidden_factor)
        self.kwargs = kwargs
        self.widen = ptn.Linear(
            in_features=self.mod_dim,
            out_features=round(hidden_factor * mod_dim),
            **kwargs
        )
        self.shrink = ptn.Linear(
            in_features=round(hidden_factor * mod_dim),
            out_features=self.mod_dim,
            **kwargs
        )

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
        return self.shrink(self.drop(self.activate(self.widen(inp))))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.drop,
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
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.gate = gate
        self.kwargs = kwargs
        self.widen = ptn.Linear(self.mod_dim, 2 * self.mod_dim, **kwargs)

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
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.gate,
            **self.kwargs
        )


class GatedHiddenBlock(Block):
    """A configurable, gated linear unit (GLU) with single hidden layer.

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
    drop: Module, optional
        Dropout to be applied after gating. Typically an instance of
        ``Dropout`` or ``AlphaDropout``. Defaults to ``Dropout(p=0.0)``,
        resulting in no dropout being applied.
    hidden_factor: int, optional
        The size of the hidden layer *before* reducing by two through gating
        is this integer factor times `mod_dim`. Defaults to 4.
    **kwargs
        Additional keyword arguments to pass through to the linear layers.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            drop: Drop = ptn.Dropout(0.0),
            hidden_factor: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.gate = gate
        self.drop = drop
        self.hidden_factor = round(hidden_factor)
        self.kwargs = kwargs
        self.widen = ptn.Linear(self.mod_dim, self.dim * 2, **kwargs)
        self.shrink = ptn.Linear(self.dim, self.mod_dim, **kwargs)

    @property
    def dim(self) -> int:
        """The hidden dimension after gating."""
        return (self.hidden_factor * self.mod_dim) // 2

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a gated linear unit (GLU) with a hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        widened = self.widen(inp)
        gated = widened[..., :self.dim] * self.gate(widened[..., self.dim:])
        return self.shrink(self.drop(gated))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.gate,
            self.drop,
            self.hidden_factor,
            **self.kwargs
        )


class ActivatedGatedBlock(Block):
    """An activated, hidden layer, followed by a gated linear unit (GLU).

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
        Dropout to be applied before gating. Typically an instance of
        ``Dropout`` or ``AlphaDropout``. Defaults to ``Dropout(p=0.0)``,
        resulting in no dropout being applied.
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
            gate: Module | Functional = ptn.Sigmoid(),
            drop: Drop = ptn.Dropout(0.0),
            hidden_factor: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.hidden_factor = round(hidden_factor)
        self.kwargs = kwargs
        self.widen = ptn.Linear(
            in_features=self.mod_dim,
            out_features=round(hidden_factor * mod_dim),
            **kwargs
        )
        self.shrink = ptn.Linear(
            in_features=round(hidden_factor * mod_dim),
            out_features=2 * self.mod_dim,
            **kwargs
        )

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a activated, hidden layer followed by a GLU.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        slim = self.shrink(self.drop(self.activate(self.widen(inp))))
        return slim[..., :self.mod_dim] * self.gate(slim[..., self.mod_dim:])

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.gate,
            self.drop,
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
        self.mod_dim = round(mod_dim)
        # Although few, some activation functions have learnable parameters
        if hasattr(activate, 'reset_parameters'):
            activate.reset_parameters()
        if hasattr(gate, 'reset_parameters'):
            gate.reset_parameters()
        self.activate = activate
        self.gate = gate
        self.drop = drop
        self.kwargs = kwargs
        self.project = ptn.Linear(self.mod_dim, self.mod_dim, **kwargs)
        self.widen = ptn.Linear(self.mod_dim, 2 * self.mod_dim, **kwargs)

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
        """Re-initialize the internal parameters of the block."""
        self.project.reset_parameters()
        self.widen.reset_parameters()
        # Although few, some activation functions have learnable parameters
        if hasattr(self.activate, 'reset_parameters'):
            self.activate.reset_parameters()
        if hasattr(self.gate, 'reset_parameters'):
            self.gate.reset_parameters()

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
    norm_first, bool, optional
        If ``True``, normalize the inputs before passing them through the block
        and adding the outputs to the raw inputs. If ``False``, pass inputs
        through the block first and normalize the sum of the inputs and outputs
        afterward. Defaults to ``True``.
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
            norm_first: bool = True,
            norm_cls: type[Module] = Identity,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.block = block
        self.drop = drop
        self.norm_first = norm_first
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
        if self.norm_first:
            return 0.5 * (inp + self.drop(self.block(self.norm(inp))))
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
            self.norm_first,
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

    Notes
    -----
    If the skip-connection sets `norm_first` to ``True``, the final output of
    the last repetition will also be normalized (with a fresh instance of the
    exact same norm type used by the skip-connection).

    """

    def __init__(self, skip: SkipConnection, n_layers: int = 2) -> None:
        super().__init__()
        self.skip = skip
        self.n_layers = n_layers
        self.blocks = ptn.Sequential(*[skip.new() for _ in self.layers])
        self.norm = skip.norm_cls(
            *skip.args, **skip.kwargs
        ) if skip.norm_first else Identity()

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
        return self.norm(self.blocks(inp))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of all blocks."""
        for block in self.blocks:
            block.reset_parameters()
        self.norm.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.skip,
            self.n_layers
        )

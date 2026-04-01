"""Flexible and composable building blocks for constructing neural-networks.

After features are embedded and combined, it is time to extract as much
information as possible to predict the desired target. One way of doing this
systematically is to repeat layers of identical internal architecture with
residual (or skip) connections between them.

"""

from typing import Any, Self
import torch as pt
import torch.nn as ptn
from .types import Module, Tensor, Functional, Block
from .misc import ResetIdentity

__all__ = [
    'ActivatedBlock',
    'ActivatedHiddenBlock',
    'GatedBlock',
    'GatedHiddenBlock',
    'GatedActivatedBlock',
    'SkipConnection',
    'Repeat'
]


class ActivatedBlock(Block):
    """A single, non-linearly activated layer.

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
    bias: bool, optional
        Whether to add a learnable bias vector in to the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            activate: Module | Functional = ptn.ELU(),
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(activate, device, dtype)
        self.bias = bias
        self.project = ptn.Linear(
            in_features=mod_dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.project.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.project.weight.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single, non-linearly activated hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.activate(self.project(inp))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.project.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(self.activate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        ActivatedBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.bias,
            self.device,
            self.dtype
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
    factor: int, optional
        The size of the hidden layer is this integer factor times `mod_dim`.
        Defaults to 4.
    bias: bool, optional
        Whether to add a learnable bias vector in to the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            activate: Module | Functional = ptn.ELU(),
            factor: int = 4,
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(activate, device, dtype)
        self.factor = factor
        self.bias = bias
        self.widen = ptn.Linear(
            in_features=mod_dim,
            out_features=factor * mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.shrink = ptn.Linear(
            in_features=factor * mod_dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.widen.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.widen.weight.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single, non-linearly activated hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.shrink(self.activate(self.widen(inp)))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(self.activate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        ActivatedHiddenBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.factor,
            self.bias,
            self.device,
            self.dtype
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
    bias: bool, optional
        Whether to add a learnable bias vector in to the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.gate = self._reset(gate, device, dtype)
        self.bias = bias
        self.widen = ptn.Linear(
            in_features=mod_dim,
            out_features=2 * mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.widen.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.widen.weight.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a single gated linear unit (GLU).

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

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
        self.gate = self._reset(self.gate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        GatedBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.gate,
            self.bias,
            self.device,
            self.dtype
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
    factor: int, optional
        The size of the hidden layer is this integer factor times `mod_dim`.
        Defaults to 4.
    bias: bool, optional
        Whether to add a learnable bias vector in to the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            gate: Module | Functional = ptn.Sigmoid(),
            factor: int = 4,
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.gate = self._reset(gate, device, dtype)
        self.factor = factor
        self.bias = bias
        self.widen = ptn.Linear(
            in_features=mod_dim,
            out_features=2 * self.dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.shrink = ptn.Linear(
            in_features=self.dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.widen.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.widen.weight.dtype

    @property
    def dim(self) -> int:
        """The hidden dimension after gating."""
        return (self.factor * self.mod_dim) // 2

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a gated linear unit (GLU) with a hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        widened = self.widen(inp)
        gated = widened[..., :self.dim] * self.gate(widened[..., self.dim:])
        return self.shrink(gated)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.widen.reset_parameters()
        self.shrink.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.gate = self._reset(self.gate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        GatedHiddenBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.gate,
            self.factor,
            self.bias,
            self.device,
            self.dtype
        )


class GatedActivatedBlock(Block):
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
        ``torch.nn.functional``, depending on whether it needs to be further
        parameterized or not. Defaults to an ``ELU`` activation.
    gate: Module or function, optional
        The activation function to be applied to half of the (non-linearly)
        projected input before multiplying with the other half. Must be
        a callable that accepts a tensor as sole argument, like a module from
        ``torch.nn`` or a function from ``torch.nn.functional``, depending
        on whether it needs to be further parameterized or not.
        Defaults to a sigmoid.
    bias: bool, optional
        Whether to add a learnable bias vector in the projections.
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.

    Note
    ----
    Inspired by Gated Residual Network (GRN) introduced in [1]_, this module
    (linearly) projects the input, applies a non-linearity, and gates the
    result by a sigmoid activation of a projection of the same intermediate
    representation, giving the model per-dimension control over how much
    non-linearity contributes to the output.

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
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(activate, device, dtype)
        self.gate = self._reset(gate, device, dtype)
        self.bias = bias
        self.project = ptn.Linear(
            in_features=mod_dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.rotate = ptn.Linear(
            in_features=mod_dim,
            out_features=mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.rotate.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.rotate.weight.dtype

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
        activated = self.activate(self.project(inp))
        gate = self.gate(self.rotate(activated))
        return gate * activated

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block."""
        self.project.reset_parameters()
        self.rotate.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(self.activate, self.device, self.dtype)
        self.gate = self._reset(self.gate, self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        GatedActivatedBlock
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.gate,
            self.bias,
            self.device,
            self.dtype
        )


class SkipConnection(Block):
    """Add a residual/skip connection around the wrapped neural-network block.

    Parameters
    ----------
    block: Block
        The block to wrap the residual/skip connection around. The reason why
        this cannot simply be a ``Module`` is that currently PyTorch does not
        provide a reasonable way of cloning them.
    dropout: float, optional
        The amount of dropout to apply to the block's output before adding it
        back to the activated residual. Defaults to 0.
    norm_first: bool, optional
        If ``True``, normalize the inputs before passing them through the block
        and adding the outputs to the raw inputs. If ``False``, pass inputs
        through the block first and normalize the sum of the inputs and outputs
        afterward. Defaults to ``True``.
    norm_cls: type, optional
        The class of the norm to be applied after adding input to output, e.g.,
        ``LayerNorm`` or ``BatchNorm1d``. Again, this is needed to easily
        create a fresh, new instances with equal, but independent parameters.
        Defaults to :class:`ResetIdentity`, resulting in no normalization.
    *args
        Arguments used to initialize an instance of `norm_cls`.
    device: str or torch.device, optional
        Torch device to first create the block on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the block in. Defaults to ``torch.float``.
    **kwargs
        Keyword arguments used to initialize an instance of `norm_cls`.

    See Also
    --------
    ~swak.pt.misc.ResetIdentity

    """

    def __init__(
            self,
            block: Block,
            dropout: float = 0.0,
            norm_first: bool = True,
            norm_cls: type[Module] = ResetIdentity,
            *args: Any,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.block = block.to(device, dtype)
        self.dropout = dropout
        self.drop = ptn.Dropout(dropout)
        self.norm_first = norm_first
        self.norm_cls: type[Module] = norm_cls
        self.args = args
        self.kwargs = kwargs
        self.norm = norm_cls(*args, device=device, dtype=dtype, **kwargs)

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.block.mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.block.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.block.dtype

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a block with the input added to the output.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        if self.norm_first:
            return inp + self.drop(self.block(self.norm(inp)))
        return self.norm(inp + self.drop(self.block(inp)))

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of the block and the norm."""
        self.block.reset_parameters()
        self.norm.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        SkipConnection
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.block.new(),
            self.dropout,
            self.norm_first,
            self.norm_cls,
            *self.args,
            device=self.device,
            dtype=self.dtype,
            **self.kwargs
        )


class Repeat(Block):
    """Repeat a skip-connection, distilling ever finer detail from your data.

    Parameters
    ----------
    skip: SkipConnection
        An instance of a :class:`SkipConnection` to repeat.
    n_layers: int, optional
        How often to repeat the `skip`. Defaults to 2.
    device: str or torch.device, optional
        Torch device to first create the blocks on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the blocks in. Defaults to ``torch.float``.

    Raises
    ------
    TypeError
        If `n_layers` is not an integer.
    ValueError
        If `n_layers` is smaller than 1.

    Note
    ----
    If the skip-connection sets `norm_first` to ``True``, no norm will be
    applied to the final output of the last repetition. If a trailing norm
    is desired, it should be applied externally, after this module.

    See Also
    --------
    SkipConnection

    """

    def __init__(
            self,
            skip: SkipConnection,
            n_layers: int = 2,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.n_layers = self.__valid(n_layers)
        self.blocks = ptn.Sequential(*[
            skip.new().to(device, dtype)
            for _ in self.layers
        ])

    @staticmethod
    def __valid(n_layers: Any) -> int:
        """Make sure `n_layers` is a positive integer."""
        try:
            as_int = int(n_layers)
        except (ValueError, TypeError) as error:
            cls = type(n_layers).__name__
            tmp = '"{}" must at least be convertible to an int, unlike {}'
            msg = tmp.format('n_layers', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) 1, unlike {}!'
            msg = tmp.format('n_layers', as_int)
            raise ValueError(msg)
        return as_int

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.blocks[0].mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.blocks[0].device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.blocks[0].dtype

    @property
    def layers(self) -> range:
        """Range of layer indices."""
        return range(self.n_layers)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass through a stack of identical skip-connection blocks.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        Tensor
            Same dimensions and sizes as the input tensor.

        """
        return self.blocks(inp)

    def reset_parameters(self) -> None:
        """Re-initialize the internal parameters of all blocks."""
        for block in self.blocks:
            block.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters.

        Returns
        -------
        Repeat
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.blocks[0],
            self.n_layers,
            self.device,
            self.dtype
        )

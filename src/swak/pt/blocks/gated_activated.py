from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Module, Tensor, Functional, Block


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

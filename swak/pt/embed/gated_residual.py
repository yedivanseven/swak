from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Tensor, Module, Functional, Drop, Block


class GatedResidualEmbedder(Block):
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
    bias: bool, optional
        Whether to add a learnable bias vector in the projection.
        Defaults to ``True``.
    drop: Module, optional
        Typically an instance of ``Dropout`` or ``AlphaDropout``. Defaults to
        ``Dropout(p=0.0)``, resulting in no dropout being applied.
    inp_dim: int, optional
        The number of features to embed together. Defaults to 1.
    device: str or pt.device, optional
        Torch device to first create the embedder on. Defaults to "cpu".
    dtype: pt.dtype, optional
        Torch dtype to first create the embedder in.
        Defaults to ``torch.float``.

    Note
    ----
    This implementation is inspired by how features are encoded in `Temporal
    Fusion Transformers`, [1]_ but it is not quite the same. Firstly, the
    (linear) projection of scalar numerical features into embedding space
    happens inside the present module. Secondly, this embedding vector is not
    transformed again (as Eq. 4 seems to imply) and there is no option to add
    a `context vector`. Thirdly, the intermediate linear layer (Eq. 3) is
    eliminated and dropout is applied to the activations after gating.
    Finally, the layer norm (Eq. 2) is replaced by simply dividing the sum of
    (linearly projected and activated) input and gated signal by 2.
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
            bias: bool = True,
            drop: Drop = ptn.Dropout(0.0),
            inp_dim: int = 1,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.bias = bias
        self.drop = drop
        self.inp_dim = inp_dim
        self.project = ptn.Linear(inp_dim, mod_dim, bias, device, dtype)
        self.widen = ptn.Linear(mod_dim, 2 * mod_dim, bias, device, dtype)
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(activate, self.device, self.dtype)
        self.gate = self._reset(gate, self.device, self.dtype)

    @property
    def mod_dim(self) -> int:
        """The embedding size."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device of all weights, biases, activations, etc. reside on."""
        return self.project.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.project.weight.dtype

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
        activated = self.activate(self.project(inp))
        wide = self.widen(activated)
        gated = wide[..., :self.mod_dim] * self.gate(wide[..., self.mod_dim:])
        return 0.5 * (activated + self.drop(gated))

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.project.reset_parameters()
        self.widen.reset_parameters()
        # Although few, some activation functions have learnable parameters
        self.activate = self._reset(self.activate, self.device, self.dtype)
        self.gate = self._reset(self.gate, self.device, self.dtype)

    def new(self) -> Self:
        """A fresh, new, re-initialized instance with identical parameters.

        Returns
        -------
        GatedResidualEmbedder
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.activate,
            self.gate,
            self.bias,
            self.drop,
            self.inp_dim,
            self.device,
            self.dtype
        )

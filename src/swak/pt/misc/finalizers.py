from typing import Self
import torch as pt
import torch.nn as ptn
from ..types import Module, Block, Tensor, Tensors, Tensors2T


class Finalizer(Block):
    """Extract one or more numbers from the final layer of a neural network.

    Instances of this class serve as a convenient final layer in any neural
    network, no matter if it is for regression, for classification, for
    multiple targets, or if you predict the parameters of a probability
    distribution. The last activations of your network are going to be passed
    through as many linear layers as you need outputs, each passed through
    their own (and potentially different) non-linearity to give you the desired
    number of outputs and the desired value range for each output.

    Parameters
    ----------
    mod_dim: int
        The size of the last dimension of the input tensor, essentially the
        "width" of the neural network before it is to be collapsed to the
        final output.
    activation: Module
        Output activation function to be applied after (linear) projection.
        Must be a ``torch.nn.Module`` and not any function from
        ``torch.nn.functional``. For unbounded regression targets, use an
        identity module.
    *activations: Module
        Additional outputs.
    bias: bool, optional
        Whether to add a learnable bias vector to the projection(s).
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the finalizer on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the finalizer in.
        Defaults to ``torch.float``.

    See Also
    --------
    Identity

    """

    def __init__(
            self,
            mod_dim: int,
            activation: Module,
            *activations: Module,
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.activations: tuple[Module, ...] = tuple(
            self._reset(act, device, dtype)
            for act in (activation, *activations)
        )
        self.bias = bias
        self.finalize = ptn.ModuleList(
            ptn.Sequential(
                ptn.Linear(
                    in_features=mod_dim,
                    out_features=1,
                    bias=bias,
                    device=device,
                    dtype=dtype
                ),
                activate
            )
            for activate in self.activations
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.finalize[0][0].weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.finalize[0][0].weight.dtype

    @property
    def n_out(self) -> int:
        """Number of final outputs."""
        return len(self.activations)

    def forward(self, inp: Tensor) -> Tensors:
        """Forward pass for extracting outputs from the final hidden layer.

        Parameters
        ----------
        inp: Tensor
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        tuple
            As many tensors are returned as there were `activations` specified.
            Each tensor has the same dimension and size, viz., that of the
            `inp` with the size of the last dimension shrunk to 1.

        """
        return tuple(finalize(inp) for finalize in self.finalize)

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        for finalize in self.finalize:
            finalize[0].reset_parameters()
            finalize[1] = self._reset(finalize[1], self.device, self.dtype)

    def new(self) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Returns
        -------
        Finalizer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            *self.activations,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype
        )


class NegativeBinomialFinalizer(Block):
    """Consistent mean and standard deviation for over-dispersed counts.

    When regressing potentially over-dispersed counts data, you might want to
    use the negative log-likelihood as loss function. However, this is only
    defined if the variance is greater than the mean. Following D. Salinas
    `et al.`, [1]_ this can be achieved by extracting a (positive) mean value
    and a (positive) scale factor from the last hidden layer of your network,
    and letting the variance be the sum of mean and the scaled square of the
    mean.

    Parameters
    ----------
    mod_dim: int
        Size of the feature space. The input tensor is expected to be of that
        size in its last dimension.
    bias: bool, optional
        Whether to add a learnable bias vector to the projection(s).
        Defaults to ``True``.
    device: str or torch.device, optional
        Torch device to first create the finalizer on. Defaults to "cpu".
    dtype: torch.dtype, optional
        Torch dtype to first create the finalizer in.
        Defaults to ``torch.float``.
    beta: float, optional
        Scaling parameter ot the `Softplus <https://pytorch.org/docs/stable/
        generated/torch.nn.Softplus.html#torch.nn.Softplus>`__ activation
        function. Defaults to 1.
    threshold: float, optional
        The `Softplus <https://pytorch.org/docs/stable/generated/
        torch.nn.Softplus.html#torch.nn.Softplus>`__ activation is approximated
        as a linear function for values greater than this. Defaults to 20.

    See Also
    --------
    ~swak.pt.losses.NegativeBinomialLoss
    ~swak.pt.dists.MuSigmaNegativeBinomial

    References
    ----------
    .. [1] D. Salinas, V. Flunkert, and J. Gasthaus, and T. Pfister, `DeepAR:
           Probabilistic Forecasting with Autoregressive Recurrent Networks`,
           `arXiv:1704.04110v3 <https://arxiv.org/pdf/1704.04110>`__ (2019).

    """

    def __init__(
            self,
            mod_dim: int,
            bias: bool = True,
            device: pt.device | str = 'cpu',
            dtype: pt.dtype = pt.float,
            beta: float = 1.0,
            threshold: float = 20.0
    ) -> None:
        super().__init__()
        self.__mod_dim = mod_dim
        self.bias = bias
        self.beta = beta
        self.threshold = threshold
        self.mu = ptn.Linear(
            in_features=mod_dim,
            out_features=1,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.alpha = ptn.Linear(
            in_features=mod_dim,
            out_features=1,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = ptn.Softplus(beta, threshold)

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.__mod_dim

    @property
    def device(self) -> pt.device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.mu.weight.device

    @property
    def dtype(self) -> pt.dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.mu.weight.dtype

    def forward(self, inp: Tensor) -> Tensors2T:
        """Forward pass for generating mean and matching standard deviation.

        Parameters
        ----------
        inp: Tensor
            The activations after the last hidden layer in your network.
            The size of the last dimension is expected to be `mod_dim`.

        Returns
        -------
        tuple
            A tensor with the predicted mean values and a tensor with the
            predicted standard deviations that are guaranteed to be greater
            or equal to the square root of the mean. Both have the same
            dimension and size, viz., that of the `inp` with the size of the
            last dimension shrunk to 1.

        """
        mu = self.activate(self.mu(inp))
        alpha = self.activate(self.alpha(inp))
        return mu, (mu * (1.0 + mu * alpha)).sqrt()

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.mu.reset_parameters()
        self.alpha.reset_parameters()

    def new(self) -> Self:
        """Return a fresh instance.

        Returns
        -------
        NegativeBinomialFinalizer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim,
            self.bias,
            self.device,
            self.dtype,
            self.beta,
            self.threshold
        )

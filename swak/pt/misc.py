from typing import Any
import torch.nn as ptn
from .types import Tensor, Module, Tensors, Tensors2T, Functional

__all__ = [
    'identity',
    'Identity',
    'Finalizer',
    'NegativeBinomialFinalizer'
]


def identity(tensor: Tensor, **_: Any) -> Tensor:
    """Simply pass through the argument and ignore keyword arguments.

    This is a placeholder for instances where a default function is required.

    Parameters
    ----------
    tensor: Tensor
        Any argument (typically a tensor) to be passed straight through.

    Returns
    -------
    Tensor
        The tensor passed in as argument.

    """
    return tensor


class Identity(Module):
    """PyTorch module that passes a tensor right through, doing nothing.

    This is a placeholder for instances where a default module is required.
    Providing any number of (keyword) arguments on instantiation is permitted,
    but they will be ignored.

    """

    def __init__(self, *_, **__) -> None:
        super().__init__()

    def forward(self, tensor: Tensor, **_: Any) -> Tensor:
        """Simply pass through the argument and ignore keyword arguments.

        Parameters
        ----------
        tensor: Tensor
            Any argument (typically a tensor) to be passed straight through.

        Returns
        -------
        Tensor
            The tensor passed in as argument.

        """
        return tensor


class Finalizer(Module):
    """Extract one or more numbers in the final layer of a neural network.

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
    *activations: Module or function
        Specify as many activation functions as you want outputs (e.g.,
        Sigmoid for binary classification, Softplus for strictly positive
        regression targets, etc.). For unbounded regression targets, where
        you want no activation function at all, use ``identity``.
    **kwargs
        Keyword arguments are passed on to all linear layers.

    """

    def __init__(
            self,
            mod_dim: int,
            *activations: Module | Functional,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.activations: tuple[Module | Functional, ...] = activations
        self.kwargs = kwargs
        self.finalize = ptn.ModuleList(
            ptn.Sequential(ptn.Linear(mod_dim, 1, **kwargs), activate)
            for activate in activations
        )

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


class NegativeBinomialFinalizer(Module):
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
    beta: float, optional
        Scaling parameter ot the `Softplus <https://pytorch.org/docs/stable/
        generated/torch.nn.Softplus.html#torch.nn.Softplus>`__ activation
        function. Defaults to 1.
    threshold: float, optional
        The `Softplus <https://pytorch.org/docs/stable/generated/
        torch.nn.Softplus.html#torch.nn.Softplus>`__ activation is approximated
        as a linear function for values greater than this. Defaults to 20.
    **kwargs
        Keyword arguments are passed on to the linear layers.


    References
    ----------
    .. [1] D. Salinas, V. Flunkert, and J. Gasthaus, and T. Pfister, `DeepAR:
           Probabilistic Forecasting with Autoregressive Recurrent Networks`,
           `arXiv:1704.04110v3 <https://arxiv.org/pdf/1704.04110>`__ (2019).

    """

    def __init__(
            self,
            mod_dim: int,
            beta: float = 1.0,
            threshold: float = 20.0,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.beta = beta
        self.threshold = threshold
        self.mu = ptn.Linear(mod_dim, 1, **kwargs)
        self.alpha = ptn.Linear(mod_dim, 1, **kwargs)
        self.map = ptn.Softplus(beta, threshold)

    def forward(self, inp: Tensor) -> Tensors2T:
        """Forward pass for generating mean and matching standard deviation.

        Parameters
        ----------
        inp: Tensor
            The activations after the last hidden layer in your network.
            The size of the last dimension is expected to be ``mod_dim``.

        Returns
        -------
        tuple
            A tensor with the predicted mean values and a tensor with the
            predicted standard deviations that are guaranteed to be greater
            or equal to the square root of the mean. Both have the same
            dimension and size, viz., that of the `inp` with the size of the
            last dimension shrunk to 1.

        """
        mu = self.map(self.mu(inp))
        alpha = self.map(self.alpha(inp))
        return mu, (mu * (1.0 + mu * alpha)).sqrt()

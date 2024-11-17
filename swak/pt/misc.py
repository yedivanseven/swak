"""Convenient classes (and functions) that do not fit any other category."""

from typing import Any, Self
import torch as pt
import torch.nn as ptn
from ..misc import ArgRepr
from .types import Tensor, Module, Tensors, Tensors2T
from .exceptions import CompileError

__all__ = [
    'identity',
    'Identity',
    'Finalizer',
    'NegativeBinomialFinalizer',
    'Compile',
    'Cat'
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

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""

    def new(self, *_, **__) -> Self:
        """Return a fresh, new instance.

        Providing any number of (keyword) arguments is permitted, but they will
        be ignored.

        Returns
        -------
        Identity
            A fresh, new instance of itself.

        """
        return self.__class__()


class Finalizer(Module):
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
    *activations: Module
        Specify as many activations (instances of PyTorch ``Module``) as you
        want outputs (e.g., ``Sigmoid()`` for binary classification,
        ``Softplus()`` for strictly positive regression targets, etc.). For
        unbounded regression targets, where you want no activation function at
        all, use ``Identity()``.
    **kwargs
        Keyword arguments are passed on to all linear layers.

    See Also
    --------
    Identity

    """

    def __init__(
            self,
            mod_dim: int,
            *activations: Module,
            **kwargs: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.activations: tuple[Module, ...] = activations
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

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        for finalize in self.finalize:
            finalize[0].reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            *activations: Module,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            The size of the last dimension of the input tensor. Overwrites the
            `mod_dim` of the current instance if given. Defaults to ``None``.
        *activations: Module
            Activation functions replace the ones in the current instance if
            any are given. If none are given, the new instance will have the
            same as the present instance.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layers together.

        Returns
        -------
        Finalizer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            *(activations or self.activations),
            **(self.kwargs | kwargs)
        )


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

    See Also
    --------
    swak.pt.losses.NegativeBinomialLoss
    swak.pt.dists.MuSigmaNegativeBinomial

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
        self.kwargs = kwargs
        self.mu = ptn.Linear(mod_dim, 1, **kwargs)
        self.alpha = ptn.Linear(mod_dim, 1, **kwargs)
        self.activate = ptn.Softplus(beta, threshold)

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
        mu = self.activate(self.mu(inp))
        alpha = self.activate(self.alpha(inp))
        return mu, (mu * (1.0 + mu * alpha)).sqrt()

    def reset_parameters(self) -> None:
        """Re-initialize all internal parameters."""
        self.mu.reset_parameters()
        self.alpha.reset_parameters()

    def new(
            self,
            mod_dim: int | None = None,
            beta: float | None = None,
            threshold: float | None = None,
            **kwargs: Any
    ) -> Self:
        """Return a fresh instance with the same or updated parameters.

        Parameters
        ----------
        mod_dim: int, optional
            The size of the last dimension of the input tensor. Overwrites the
            `mod_dim` of the current instance if given. Defaults to ``None``.
        beta: float, optional
            Scaling parameter ot the ``Softplus`` activation function.
            Overwrites the `beta` of the current instance if given.
            Defaults to ``None``.
        threshold: float, optional
            The ``Softplus`` activation is approximated as a linear function
            for values greater than this. Overwrites the `threshold` of the
            current instance if given. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into the keyword arguments
            of the current instance and are then passed through to the linear
            layers together.

        Returns
        -------
        NegativeBinomialFinalizer
            A fresh, new instance of itself.

        """
        return self.__class__(
            self.mod_dim if mod_dim is None else mod_dim,
            self.beta if beta is None else beta,
            self.threshold if threshold is None else threshold,
            **(self.kwargs | kwargs)
        )


class Compile:
    """Partial of the ``compile`` top-level function or Module method.

    Parameters
    ----------
    inplace: bool, optional
        Whether to compile the model in place (by calling its ``compile``
        method) or create a new, compiled instance. Defaults to ``True``
    model: Module, optional
        For convenience, the model to compile can already be given at
        instantiation. However, Nothing will happen until instances are called.
    **kwargs
        Additional keyword arguments are forwarded to the ``compile`` function
        or method call. See the `Documentation <https://pytorch.org/docs/
        stable/generated/torch.compile.html#torch-compile>`__ for details.

    """

    def __init__(
            self,
            inplace: bool = True,
            model: Module | None = None,
            **kwargs: Any
    ) -> None:
        self.inplace = inplace
        self.model = model
        self.kwargs = kwargs

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        model = None if self.model is None else 'model'
        kwargs = (f'{k}={str(v)}' for k, v in self.kwargs.items())
        kwargs = ', '.join(kwargs)
        sep = ', ' if kwargs else ''
        return f'{cls}({self.inplace}, {model}{sep}{kwargs})'

    def __call__(self, model: Module | None = None, **kwargs: Any) -> Module:
        """Compile a Module with the given options.

        Parameters
        ----------
        model: Module, optional
            If no model was given on instantiation, one must be given here.
            Otherwise, there would be nothing to compile. If a model was given
            on instantiation and one is given here, the latter replaces the
            former. Defaults to ``None``.
        **kwargs
            Additional keyword arguments are merged into those given at
            instantiation and then forward to the ``compile`` function
            or method call. See the `Documentation <https://pytorch.org/docs/
            stable/generated/torch.compile.html#torch-compile>`__ for details.

        Returns
        -------
        Module
            The compiled module.

        Raises
        ------
        CompileError
            If no model was given, neither at instantiation, nor when calling
            instances.

        """
        model = self.model if model is None else model
        if model is None:
            raise CompileError('No model to compile!')
        merged_kwargs = self.kwargs | kwargs
        if self.inplace:
            model.compile(**merged_kwargs)
            return model
        return pt.compile(model, **merged_kwargs)


class Cat(ArgRepr):
    """Simple partial of PyTorch's top-level `cat` function.

    Parameters
    ----------
    dim: int, optional
        The dimension along which to concatenate the tensors. Defaults to 0.

    """

    def __init__(self, dim: int = 0) -> None:
        super().__init__(dim)
        self.dim = dim

    def __call__(self, tensors: Tensors | list[Tensor]) -> Tensor:
        """Concatenate the given tensors along one of their dimensions.

        Parameters
        ----------
        tensors: tuple or list of tensors
            Tensors to concatenate along an existing dimension.

        Returns
        -------
        Tensor
            The concatenated tensors.

        """
        return pt.cat(tensors, dim=self.dim)

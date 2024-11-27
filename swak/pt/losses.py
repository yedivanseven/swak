"""Custom loss functions not (yet) shipped with PyTorch  out of the box.

Especially when networks should make probabilistic predictions, that is, when
they are set up to predict the parameters of an analytical probability mass or
density function instead of just an expectation value, special loss functions
are required that implement the negative log-likelihood of the matching
probability distribution.

"""

from typing import Self
from enum import StrEnum
from typing import Literal
import math
import torch as pt
import torch.nn as ptn
import torch.special as pts
from .types import Tensor, Module
from .exceptions import LossError
from .misc import identity

__all__ = [
    'Reduction',
    'RMSELoss',
    'TweedieLoss',
    'BetaBernoulliLoss',
    'GammaLoss',
    'StudentLoss',
    'NegativeBinomialLoss',
    'XEntropyLoss'
]

type LiteralReduction = Literal['mean', 'sum', 'none']


class Reduction(StrEnum):
    """Specify the aggregation level when evaluating loss functions."""
    MEAN = 'mean'
    SUM = 'sum'
    NONE = 'none'


class _BaseLoss(Module):
    """Base class for custom loss functions.

    This is class is not meant to ever be instantiated directly. Rather,
    it is intended to be inherited from when implementing a custom loss
    function. Its purpose is mimicking the call signature of PyTorch builtin
    loss functions so that only the `forward` method needs to be implemented
    in subclasses.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    See Also
    --------
    Reduction

    """

    __reductions = {
        'mean': pt.mean,
        'sum': pt.sum,
        'none': identity
    }

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.reduction = str(reduction).strip().lower()
        self.register_buffer('eps', pt.tensor(eps), False)
        try:
            self._reduce = self.__reductions[self.reduction]
        except KeyError:
            msg = '"reduction" must be one of "mean", "sum", or "none"!'
            raise LossError(msg)

    def forward(self, *tensors: Tensor) -> Tensor:
        """Forward pass for custom losses. Implement in subclasses!"""
        raise NotImplementedError('Subclasses must implement a forward pass!')


class RMSELoss(Module):
    """Root mean squared error loss.

    PyTorch only comes with a mean squared error loss. Since it is often more
    intuitive to compare error and target value when they are on the same
    scale, the square root of the MSE is naively implemented here.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-8.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.register_buffer('eps', pt.tensor(eps), False)
        self.mse = ptn.MSELoss(reduction=reduction)

    def forward(self, y_hat: Tensor, y_true: Tensor) -> Tensor:
        """Compute RMSE loss between predicted and observed values.

        For numerical stability, MSE values smaller than `eps` will be
        clamped to `eps` before taking the square root.

        Parameters
        ----------
        y_hat: Tensor
            Predicted expectation values.
        y_true: Tensor
            Actually observed values. Should have the same dimensionality
            as `y_hat`, but this will not be checked for, let alone enforced.

        Returns
        -------
        Tensor
            The root mean square error.

        """
        return self.mse(y_hat, y_true).clamp(self.eps).sqrt()


class TweedieLoss(_BaseLoss):
    """Tweedie loss for zero-inflated, right-skewed, long-tailed target.

    Implements the `deviance <https://en.wikipedia.org/wiki/Deviance
    _(statistics)>`__ of the `Tweedie distribution <https://en.wikipedia.org/
    wiki/Tweedie_distribution>`__ for power parameters between 1 and 2, where
    it can be seen as a compound Poisson-Gamma distribution.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__(reduction, eps)

    def forward(self, mu: Tensor, p: Tensor, y_true: Tensor) -> Tensor:
        """Forward pass for the Tweedie loss function.

        Parameters
        ----------
        mu: Tensor
            Predicted expectation values. Should be greater than or equal
            to 0, but this will not be checked for, let alone enforced.
            However, `eps` will be added to ensure numerical stability.
        p: Tensor
            Predicted values for the power parameter. Should have values
            strictly between 1 and 2. All values beyond that range will be
            clamped to lie within this range (by at least `eps`).
        y_true: Tensor
            Actually observed values. Should be greater than or equal to 0,
            but this will not be checked for, let alone enforced.

        Returns
        -------
        Tensor
            Proportional to the negative log-likelihood of a Tweedie
            distribution for 1 < `p` < 2.

        """
        p_ = p.clamp(1.0 + self.eps, 2.0 - self.eps)
        mu_ = mu + self.eps

        negll = - 2.0 * (
            y_true * mu_.pow(1.0 - p_) / (1.0 - p_) -
            y_true.pow(2.0 - p_) / (1.0 - p_) / (2.0 - p_) -
            mu_.pow(2.0 - p_) / (2.0 - p_)
        )
        return self._reduce(negll)


class BetaBernoulliLoss(_BaseLoss):
    """Special case of the Beta-Binomial negative log-likelihood for 1 trial.

    Use to make probabilistic predictions for binary classification, where
    the model outputs the alpha and beta coefficients of the Beta-distributed
    success probability instead of a point estimate of the success probability.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__(reduction, eps)

    def forward(self, alpha: Tensor, beta: Tensor, y_true: Tensor) -> Tensor:
        """Forward pass for the Beta-Binomial loss function.

        Parameters
        ----------
        alpha: Tensor
            Predicted values for the `alpha` parameter of the Beta-distributed
            success probability in binary classification. Should be greater
            than or equal to 0, but this will not be checked for, let alone
            enforced. However, `eps` is added to ensure numerical stability.
        beta: Tensor
            Predicted values for the `beta` parameter of the Beta-distributed
            success probability in binary classification. Should be greater
            than or equal to 0, but this will not be checked for, let alone
            enforced. However, `eps` is added to ensure numerical stability.
        y_true: Tensor
            Actually observed binary outcomes, encoded as 1.0 for success and
            0.0 for failure.

        Returns
        -------
        Tensor
            Negative log-likelihood of the Beta-Binomial distribution for the
            special case of one trial.

        """
        alpha_ = alpha + self.eps
        beta_ = beta + self.eps

        negll = -(
            pts.gammaln(alpha_ + y_true) +
            pts.gammaln(beta_ + (1.0 - y_true)) -
            pts.gammaln(alpha_) -
            pts.gammaln(beta_) -
            (alpha + beta + self.eps).log()
        )
        return self._reduce(negll)


class GammaLoss(_BaseLoss):
    """Negative log-likelihood of a scaled Gamma distribution.

    Potentially useful for strictly positive targets and skewed residuals.
    For convenience, the Gamma distribution is parameterized in terms of
    mean and standard deviation instead of its standard form in terms of
    shape and scale (or rate) parameters.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__(reduction, eps)

    def forward(self, mu: Tensor, sigma: Tensor, y_true: Tensor) -> Tensor:
        """Forward pass for the scaled Gamma loss function.

        Parameters
        ----------
        mu: Tensor
            Predicted mean values. Should be greater than or equal to 0, but
            this will not be checked for, let alone enforced. However, `eps`
            is added to ensure numerical stability.
        sigma: Tensor
            Predicted standard deviations. Should be greater than or equal
            to 0, but this will not be checked for, let alone enforced.
            However, `eps` is added to ensure numerical stability.
        y_true: Tensor
            Actually observed values. Should be greater than or equal
            to 0, but this will not be checked for, let alone enforced.
            However, `eps` is added to ensure numerical stability.

        Returns
        -------
        Tensor
            The negative log-likelihood of a scaled Gamma distribution.

        """
        mu_ = mu + self.eps
        sigma2 = sigma.pow(2.0) + self.eps
        ratio = mu_.pow(2.0) / sigma2
        y_true_ = y_true + self.eps
        scaled = y_true * mu / sigma2 + self.eps

        negll = -(
            ratio * scaled.log() -
            y_true_.log() -
            scaled -
            pts.gammaln(ratio)
        )
        return self._reduce(negll)


class StudentLoss(_BaseLoss):
    """Negative log-likelihood of a non-standardized Student's t distribution.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__(reduction, eps)
        self.register_buffer('const', -0.5 * pt.tensor(math.pi).log(), False)

    def forward(
            self,
            df: Tensor,
            loc: Tensor,
            scale: Tensor,
            y_true: Tensor
    ) -> Tensor:
        """Forward pass for the Student's t loss function.

        Parameters
        ----------
        df: tensor
            Predicted degrees of freedom. Should be greater than zero, but this
            will not be checked for, let alone enforced. However, `eps` is
            added for numerical stability.
        loc: Tensor
            Predicted mean values.
        scale: Tensor
            Predicted scales. Should be greater than or equal to 0, but this
            will not be checked for, let alone enforced. However, `eps` is
            added to ensure numerical stability.
        y_true:
            Actually observed values.

        Returns
        -------
        Tensor
            Negative log-likelihood of a Student's t distribution.

        """
        half_nu = 0.5 * df + self.eps
        half_nu1p = 0.5 * (df + 1.0)
        tau2nu = scale.pow(2.0) * df + self.eps

        negll = -(
            self.const +
            pts.gammaln(half_nu1p) -
            pts.gammaln(half_nu) -
            0.5 * tau2nu.log() -
            half_nu1p * ((y_true - loc).pow(2.0) / tau2nu).log1p()
        )
        return self._reduce(negll)


class NegativeBinomialLoss(_BaseLoss):
    """Negative log-likelihood of a Negative-Binomial distribution.

    Potentially useful for counts data where more flexibility than provided
    by a Poisson loss is desired because the data might be over-dispersed.
    For convenience, the Negative-Binomial distribution is parameterized in
    terms of mean and standard deviation instead of its standard form.

    Parameters
    ----------
    reduction: string, optional
        One of "mean", "sum" or "none". Defaults to "mean". Whether and, if so,
        how to aggregate the tensor resulting from evaluating the point-wise
        loss function on the input. Use the ``Reduction`` enum to avoid typos.
    eps: float, optional
        Many loss functions require input and/or target tensors to be bound
        by some lower and/or upper value. It is the user's responsibility to
        ensure that they are. However, evaluating a loss function just at the
        interval boundary of its support might lead to numerical inaccuracies.
        To avoid these, it is often advisable to shift such values away from
        boundaries by a small value `eps`. Defaults to 1e-6.

    Note
    ----
    The present parameterization only makes sense if the variance is strictly
    greater than the mean. This is best taken into account already on the
    model side, e.g., by forcing the output for `sigma` to be a that of `mu`
    multiplied by one plus some (learnable) fraction of `mu`, as described
    by D. Salinas `et al.` in their DeepAR paper. [1]_

    See Also
    --------
    Reduction
    swak.pt.misc.NegativeBinomialFinalizer
    swak.pt.dists.MuSigmaNegativeBinomial

    References
    ----------
    .. [1] D. Salinas, V. Flunkert, and J. Gasthaus, and T. Pfister, `DeepAR:
           Probabilistic Forecasting with Autoregressive Recurrent Networks`,
           `arXiv:1704.04110v3 <https://arxiv.org/pdf/1704.04110>`__ (2019).

    """

    def __init__(
            self,
            reduction: LiteralReduction = 'mean',
            eps: float = 1e-6
    ) -> None:
        super().__init__(reduction, eps)


    def forward(self, mu: Tensor, sigma: Tensor, y_true: Tensor) -> Tensor:
        """Forward pass for the Negative-Binomial loss function.

        Parameters
        ----------
        mu: Tensor
            Predicted mean values. Should be greater than or equal to 0, but
            this will not be checked for, let alone enforced. However, `eps`
            is added to ensure numerical stability.
        sigma: Tensor
            Predicted standard deviations. Should be greater than or equal
            to the square root of `mu`, but this will not be checked for,
            let alone enforced. However, `eps` is added to ensure numerical
            stability.
        y_true: Tensor
            Actually observed values. Should be greater than or equal
            to 0, but this will not be checked for, let alone enforced.

        Returns
        -------
        Tensor
            The negative log-likelihood of a Negative-Binomial distribution.

        """
        sigma2 = sigma.pow(2.0) + 2 * self.eps
        scaled = mu / sigma2 + self.eps
        ratio = mu.pow(2.0) / (sigma2 - mu) + self.eps

        negll = -(
            pts.gammaln(y_true + ratio) +
            y_true * (1.0 - scaled).log() +
            ratio * scaled.log() -
            pts.gammaln(y_true + 1.0) -
            pts.gammaln(ratio)
        )
        return self._reduce(negll)


class XEntropyLoss(ptn.CrossEntropyLoss):
    """Subclass of PyTorch's ``CrossEntropyLoss``  with added functionality.

    When in training mode (toggled by calling the module ``train`` method
    without argument or with ``True``), the `label_smoothing` is applied
    according to value provided at instantiation. When in evaluation mode,
    however, the `label-smoothing` is set to 0.0 to report reproducible
    and unbiased values for test- or validation loss and (log-)perplexity.

    Parameters
    ----------
    weight: Tensor, optional
        A manual rescaling weight given to each class. If given, has to be a
        1D-tensor of a size equal to the number of classes and a floating point
        dtype. Defaults to ``None``.
    ignore_index: int, optional
        Specifies a target value that is ignored and does not contribute to
        the input gradient. When `reduction` is "mean"", the loss is averaged
        only over non-ignored targets. Only applicable when the target contains
        class indices. Defaults to -100.
    reduction: str, optional
        Specifies the reduction to apply to the output: "none", "mean", or
        "sum". Defaults to "mean", which means the weighted mean in case a
        valid `weight` was provided. Use the ``Reduction`` enum to avoid typos.
    label_smoothing: float, optional
         Specifies the amount of smoothing when computing the loss. Must lie
         in the interval [0.0, 1.0] where 0.0 means no smoothing. The targets
         become a mixture of the original ground truth and a uniform
         distribution as described in `Rethinking the Inception Architecture
         for Computer Vision <https://arxiv.org/abs/1512.00567>`__.
         Default to 0.0

    Note
    ----
    For more information on this loss, please refer to the full PyTorch
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.
    CrossEntropyLoss.html#crossentropyloss>`__.

    See Also
    --------
    Reduction

    """

    def __init__(
            self,
            weight: Tensor | None = None,
            ignore_index: int = -100,
            reduction: Reduction | LiteralReduction = 'mean',
            label_smoothing: float = 0.0
    ) -> None:
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=str(reduction).strip().lower(),
            label_smoothing=label_smoothing
        )
        self.__cached_label_smoothing = self.label_smoothing

    def train(self, mode: bool = True) -> Self:
        """Toggle training mode: `label-smoothing` as given at instantiation.

        Parameters
        ----------
        mode: bool, optional
            Whether to switch training mode on (``True``) or off (``False``).
            Defaults to ``True``.

        Returns
        -------
        XEntropyLoss
            Itself in evaluation mode.

        """
        self.label_smoothing = self.__cached_label_smoothing if mode else 0.0
        return super().train(mode)

    def eval(self) -> Self:
        """Toggle evaluation mode by setting `label-smoothing` to 0.0

        Calling ``eval`` is Equivalent to calling the method ``train(False)``.

        Returns
        -------
        XEntropyLoss
            Itself in evaluation mode.

        """
        self.label_smoothing = 0.0
        return super().eval()

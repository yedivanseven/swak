"""PyTorch probability distributions re-parameterized for convenience.

The output of networks designed to provide probabilistic predictions becomes
much more interpretable when they predict the mean and the standard deviation
of some probability distribution rather than its (often arcance) natural
parameters. To that end, this module provides re-parameterized versions
of probability distributions to match the parameterization of the respective
log-likelihoods implemented as loss functions.

"""

import torch.distributions as ptd
from .types import Tensor

__all__ = [
    'MuSigmaGamma',
    'MuSigmaNegativeBinomial',
    'MeanStdLogNormal',
]


class MuSigmaGamma(ptd.Gamma):
    """Subclass of the :class:`Gamma` distribution with intuitive parameters.

    Parameters
    ----------
    mu: Tensor
        Mean value(s) of the Gamma distribution(s).
    sigma: Tensor
        Standard deviation(s) of the Gamma distribution(s).
    validate_args: bool, optional
        Whether the parent class should validate the transformed parameters
        or not. Defaults to ``False``

    See Also
    --------
    ~swak.pt.losses.GammaLoss

    """

    def __init__(
            self,
            mu: float | Tensor,
            sigma: float | Tensor,
            validate_args: bool = False
    ) -> None:
        variance = sigma**2
        super().__init__(mu**2 / variance, mu / variance, validate_args)


class MuSigmaNegativeBinomial(ptd.NegativeBinomial):
    """Subclass of the PyTorch :class:`NegativeBinomial` distribution.

    Parameters
    ----------
    mu: Tensor
        Mean value(s) of the Gamma distribution(s).
    sigma: Tensor
        Standard deviation(s) of the Gamma distribution(s).
    validate_args: bool, optional
        Whether the parent class should validate the transformed parameters
        or not. Defaults to ``False``

    Note
    ----
    This parameterization only makes sense if the variance is strictly
    greater than the mean. This is best taken into account already on the
    model output side, but can be checked here by setting `validate_args`
    to ``True``.

    See Also
    --------
    ~swak.pt.losses.NegativeBinomialLoss
    ~swak.pt.misc.NegativeBinomialFinalizer

    """

    def __init__(
            self,
            mu: float | Tensor,
            sigma: float | Tensor,
            validate_args: bool = False
    ) -> None:
        variance = sigma**2
        super().__init__(
            mu**2 / (variance - mu),
            1.0 - mu / variance,
            validate_args=validate_args
        )


class MeanStdLogNormal(ptd.LogNormal):
    """Subclass of the PyTorch :class:`LogNormal` distribution.

    Parameters
    ----------
    mean: Tensor
        Mean value(s) of the Log-Normal distribution(s) in natural scale.
    std: Tensor
        Standard deviation(s) of Log-Normal distribution(s) in natural scale.
    validate_args: bool, optional
        Whether the parent class should validate the transformed parameters
        or not. Defaults to ``False``

    See Also
    --------
    ~swak.pt.losses.LogNormalLoss

    """

    def __init__(
            self,
            mean: Tensor,
            std: Tensor,
            validate_args: bool = False
    ) -> None:
        variance = (std.pow(2) / mean.pow(2)).log1p()
        super().__init__(
            mean.log() - 0.5 * variance,
            variance.sqrt(),
            validate_args=validate_args
        )

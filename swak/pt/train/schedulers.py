from typing import Any
from functools import cached_property
from torch.nn.modules.module import _IncompatibleKeys
from ..types import Optimizer
from ...misc import ArgRepr

__all__ = [
    'NoSchedule',
    'LinearInverse',
    'LinearExponential'
]


class NoSchedule(ArgRepr):
    """Mock learning-rate scheduler that does not actually do anything.

    This class is just a placeholder for an actual PyTorch learning-rate
    scheduler. Any additional (keyword) arguments provided at instantiation
    are ignored.

    Parameters
    ----------
    optimizer: Optimizer
        A PyTorch optimizer instance.

    """

    def __init__(
            self,
            optimizer: Optimizer,
            *_: Any,
            **__: Any
    ) -> None:
        super().__init__(optimizer)
        self.optimizer = optimizer

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Does not do anything, certainly not touching the learning rate.

        Positional and/or keyword arguments can be provided, but are ignored.

        """

    def get_last_lr(self) -> list[float]:
        """Returns the default learning rate of the optimizer."""
        return [self.optimizer.defaults['lr']]

    def state_dict(self) -> dict[str, Any]:
        """Because there is no state return an empty, ordered dictionary."""
        return {}

    def load_state_dict(self, *_: Any, **__: Any) -> _IncompatibleKeys:
        """Do nothing, return the object thing as other methods of that type"""
        return _IncompatibleKeys([], [])


class LinearInverse(ArgRepr):
    """Scale up learning rate during warmup before decaying with inverse power.

    Instances of this class are not learning-rate schedulers by themselves!
    They are intended to be passed as ``lr_lambda`` argument to PyTorch's
    `LambdaLR <https://pytorch.org/docs/stable/generated/torch.optim.
    lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR>`__
    learning-rate scheduler.

    Parameters
    ----------
    warmup: int, optional
        Number of epochs during which the learning rate will be linearly
        scaled up to the one specified in the optimizer. Defaults to 1,
        resulting in the learning rate already at its maximum value in the
        first epoch.
    power: float, optional
        After `warmup` epochs, the learning rate is scaled down with the
        inverse epoch number taken to the power of this number. Values are
        cut to lie in the interval [0.5, 1.0].
        Defaults to 0.5, the slowest decay.

    """

    def __init__(self, warmup: int = 1, power: float = 0.5) -> None:
        super().__init__(warmup, power)
        self.warmup = max(warmup, 1)
        self.power = min(max(power, 0.5), 1.0)

    @cached_property
    def ramp(self) -> list[float]:
        """Learning-rate scaling factors during the warmup period."""
        return [epoch / self.warmup for epoch in range(1, self.warmup + 1)]

    def __call__(self, epoch: int) -> float:
        """Learning rate scaling factor depending on the epoch.

        Parameters
        ----------
        epoch: int
            Epoch to return the learning-rate scaling factor for.

        Returns
        -------
        float
            The learning-rate scaling factor

        """
        if epoch < self.warmup:
            return self.ramp[epoch]
        return (2 + (epoch - self.warmup)) ** -self.power


class LinearExponential(ArgRepr):
    """Scale up learning rate during warmup before decaying with inverse power.

    Instances of this class are not learning-rate schedulers by themselves!
    They are intended to be passed as ``lr_lambda`` argument to PyTorch's
    `LambdaLR <https://pytorch.org/docs/stable/generated/torch.optim.
    lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR>`__
    learning-rate scheduler.

    Parameters
    ----------
    warmup: int, optional
        Number of epochs during which the learning rate will be linearly
        scaled up to the one specified in the optimizer. Defaults to 1,
        resulting in the learning rate already at its maximum value in the
        first epoch.
    gamma: float, optional
        After `warmup` epochs, the learning rate is scaled down with this
        number to the power of the epoch number. Therefore, it must lie in
        the interval (0.0, 1.0). Defaults to 0.95.

    """

    def __init__(self, warmup: int = 1, gamma: float = 0.95) -> None:
        super().__init__(warmup, gamma)
        self.warmup = max(warmup, 1)
        self.gamma = min(max(gamma, 0.0), 1.0)

    @cached_property
    def ramp(self) -> list[float]:
        """Learning-rate scaling factors during the warmup period."""
        return [epoch / self.warmup for epoch in range(1, self.warmup + 1)]

    def __call__(self, epoch: int) -> float:
        """Learning rate scaling factor depending on the epoch.

        Parameters
        ----------
        epoch: int
            Epoch to return the learning-rate scaling factor for.

        Returns
        -------
        float
            The learning-rate scaling factor

        """
        if epoch < self.warmup:
            return self.ramp[epoch]
        return self.gamma ** (1 + epoch - self.warmup)

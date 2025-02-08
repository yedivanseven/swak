import math
from typing import Any
from torch.nn.modules.module import _IncompatibleKeys
from ..types import Optimizer
from ...misc import ArgRepr

__all__ = [
    'NoSchedule',
    'LinearInverse',
    'LinearExponential',
    'LinearCosine'
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
        """Because there is no state, return an empty dictionary."""
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
        Number of steps during which the learning rate will be linearly
        scaled up to the one specified in the optimizer. Defaults to 0,
        resulting in the learning rate already at its maximum value for the
        first step and only decaying thereafter.
    power: float, optional
        After `warmup` steps, the learning rate is scaled down with the
        inverse step number taken to the power of this number. Values are
        cut to lie in the interval [0.5, 1.0].
        Defaults to 0.5, the slowest decay.

    """

    def __init__(self, warmup: int = 0, power: float = 0.5) -> None:
        super().__init__(warmup, power)
        self.warmup = max(warmup, 0)
        self.power = min(max(power, 0.5), 1.0)

    def __call__(self, step: int) -> float:
        """Learning rate scaling factor depending on the step.

        Parameters
        ----------
        step: int
            Step to return the learning-rate scaling factor for.

        Returns
        -------
        float
            The learning-rate scaling factor

        """
        if step < self.warmup:
            return (step + 1) / self.warmup
        if step == self.warmup:
            return 1.0
        return (step - self.warmup) ** -self.power


class LinearExponential(ArgRepr):
    """Scale up learning rate during warmup before decaying exponentially.

    Instances of this class are not learning-rate schedulers by themselves!
    They are intended to be passed as ``lr_lambda`` argument to PyTorch's
    `LambdaLR <https://pytorch.org/docs/stable/generated/torch.optim.
    lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR>`__
    learning-rate scheduler.

    Parameters
    ----------
    warmup: int, optional
        Number of steps during which the learning rate will be linearly
        scaled up to the one specified in the optimizer. Defaults to 0,
        resulting in the learning rate already at its maximum value for the
        first step and only decaying thereafter.
    gamma: float, optional
        After `warmup` steps, the learning rate is scaled down with this
        number to the power of the step number. Therefore, it must lie in
        the interval (0.0, 1.0). Defaults to 0.95.

    """

    def __init__(self, warmup: int = 0, gamma: float = 0.95) -> None:
        super().__init__(warmup, gamma)
        self.warmup = max(warmup, 0)
        self.gamma = min(max(gamma, 0.0), 1.0)

    def __call__(self, step: int) -> float:
        """Learning rate scaling factor depending on the step.

        Parameters
        ----------
        step: int
            Step to return the learning-rate scaling factor for.

        Returns
        -------
        float
            The learning-rate scaling factor

        """
        if step < self.warmup:
            return (step + 1) / self.warmup
        return self.gamma ** (step - self.warmup)


class LinearCosine(ArgRepr):
    """Scale up learning rate during warmup before decaying with cosine.

    Instances of this class are not learning-rate schedulers by themselves!
    They are intended to be passed as ``lr_lambda`` argument to PyTorch's
    `LambdaLR <https://pytorch.org/docs/stable/generated/torch.optim.
    lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR>`__
    learning-rate scheduler.

    Parameters
    ----------
    warmup: int, optional
        Number of steps during which the learning rate will be linearly
        scaled up to the one specified in the optimizer. Defaults to 0,
        resulting in the learning rate already at its maximum value for the
        first step and only decaying thereafter.
    cooldown: int, optional
        After `warmup` steps, the learning rate is scaled down with a cosine
        function for this many steps. Defaults to 100, but must be at least 1.

    Notes
    -----
    The learning rate will never actually reach 0. Rather, it stays at the last
    value right before reaching 0. How small this value is, depends on the
    choice for `max_steps`. For `max_steps` = 1, it will stay at 1.0, for
    `max_steps` = 2, it will stay at 0.5, and so on.

    """

    def __init__(
            self,
            warmup: int = 0,
            cooldown: int = 100
    ) -> None:
        super().__init__(warmup, cooldown)
        self.warmup = max(warmup, 0)
        self.cooldown = max(1, cooldown)

    def __call__(self, step: int) -> float:
        """Learning rate scaling factor depending on the step.

        Parameters
        ----------
        step: int
            Step to return the learning-rate scaling factor for.

        Returns
        -------
        float
            The learning-rate scaling factor

        """
        if step < self.warmup:
            return (step + 1) / self.warmup
        if step < (self.warmup + self.cooldown):
            angle = math.pi * (step - self.warmup) / self.cooldown
            return 0.5 + 0.5 * math.cos(angle)
        angle = math.pi * (self.cooldown - 1) / self.cooldown
        return 0.5 + 0.5 * math.cos(angle)

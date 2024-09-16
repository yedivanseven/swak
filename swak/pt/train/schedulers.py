from typing import Any
from ..types import Optimizer
from ...magic import ArgRepr


class NoSchedule(ArgRepr):
    """Mock learning-rate scheduler that does not actually do anything.

    Parameters
    ----------
    optimizer
        A PyTorch optimizer instance.

    """

    def __init__(
            self,
            optimizer: Optimizer,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.optimizer = optimizer

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Does not do anything, certainly not touching the learning rate."""

    def get_last_lr(self) -> list[float]:
        """Returns the default learning rate of the optimizer."""
        return [self.optimizer.defaults['lr']]

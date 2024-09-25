from typing import Any
from torch.nn.modules.module import _IncompatibleKeys
from ..types import Optimizer
from ...misc import ArgRepr

__all__ = [
    'NoSchedule'
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

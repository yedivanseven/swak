from typing import TypedDict, Any
from abc import ABC, abstractmethod
from collections.abc import Callable
from ...misc import ArgRepr
from ..types import Module, Batches

__all__ = [
    'EpochCallback',
    'EpochPrinter',
    'TrainCallback',
    'History',
    'TrainPrinter'
]

type TrainCallback = Callable[[int, int, float, bool, History], None]


class History(TypedDict):
    """Summary of metrics passed to the callback when training is finished."""
    train_loss: list[float]  #: List of losses evaluated on train data.
    test_loss: list[float | None]  #: List of losses evaluated on test data.
    lr: list[float]  #: List of learning rates.


class EpochCallback(ABC):
    """Base class to inherit from when implementing custom epoch callbacks."""

    @abstractmethod
    def __call__(
            self,
            epoch: int,
            train_loss: int,
            test_loss: int,
            learning_rate: float,
            model: Module,
            data: Batches,
    ) -> None:
        """Called after each epoch to print, log, or otherwise analyse metrics.

        Parameters
        ----------
        epoch: int
            The current epoch in the training loop.
        train_loss: float
            The loss computed on a sample of the train data after the
            current `epoch`.
        test_loss: float
            The loss computed on a sample of the test data after the
            current `epoch`. Always ``nan`` if no test data is used.
        learning_rate: float
            Learning rate of the optimizer in the current `epoch`.
        model: Module
            A reference to the model being trained.
        data: Batches
            An iterator over 2-tuples of a feature-tensor tuple and a target
            tensor to call the `model` with. Sampled from the test data if
            present and from the train data otherwise.

        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Called at the end of training to be compatible with TensorBoard."""
        ...


class EpochPrinter(ArgRepr, EpochCallback):
    """Epoch callback assembling an informative message on training progress.

    Parameters
    ----------
    printer: callable, optional
        Will be called with the assembled message. Defaults to the python
        builtin ``print`` function, but could also be a logging command.

    """

    def __init__(self, printer: Callable[[str], Any] = print) -> None:
        super().__init__(printer)
        self.printer = printer

    def __call__(
            self,
            epoch: int,
            train_loss: float,
            test_loss: float,
            learning_rate: float,
            model: Module,
            data: Batches
    ) -> None:
        """Assemble training-progress message and call the printer with it."""
        msg = (f'Epoch: {epoch:>3} | learning rate: {learning_rate:7.5f} | '
               f'train loss: {train_loss:7.5f} | test loss: {test_loss:7.5f}')
        self.printer(msg)

    def close(self) -> None:
        """Does nothing because there is nothing to close."""


class TrainPrinter(ArgRepr):
    """Train callback assembling an informative message when training ends.

    Parameters
    ----------
    printer: callable, optional
        Will be called with the assembled message. Defaults to the python
        builtin ``print`` function, but could also be a logging command.

    """
    def __init__(self, printer: Callable[[str], Any] = print) -> None:
        super().__init__(printer)
        self.printer = printer

    def __call__(
            self,
            epoch: int,
            best_epoch: int,
            best_loss: float,
            max_epochs_reached: bool,
            history: History
    ) -> None:
        """Assemble and print a summary of model training.

        Parameters
        ----------
        epoch: int
            The last epoch in the training loop.
        best_epoch: int
            The epoch with the lowest loss encountered.
        best_loss: float
            The lowest loss encountered.
        max_epochs_reached: bool
            Whether the maximum number of epochs was exhausted or not.
        history: History
            Dictionary with lists of train losses, test losses, and learning
            rates.

        """
        if max_epochs_reached:
            self.printer(f'Maximum number of {epoch} epochs exhausted!')
        else:
            msg = (f'Stopping after {epoch} epochs because, even after waiting'
                   f' for {epoch - best_epoch} more epochs, the loss did not '
                   f'drop below the lowest value of {best_loss:7.5f} seen in '
                   f'epoch {best_epoch}. Recovering that checkpoint.')
            self.printer(msg)

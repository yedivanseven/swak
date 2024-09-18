import copy
from typing import TypedDict, Any
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from ...magic import ArgRepr
from ..types import Tensor, Tensors, Module, Optimizer, LRScheduler

type TrainCallback = Callable[[int, int, float, bool, History], None]


class History(TypedDict):
    train_loss: list[float]
    test_loss: list[float | None]
    lr: list[float]


class State(TypedDict):
    epoch: int
    loss: float
    model: OrderedDict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]

# ToDo: Write unit tests and docstrings!
class EpochCallback(ABC):

    @abstractmethod
    def __call__(
            self,
            epoch: int,
            train_loss: int,
            test_loss: int,
            learning_rate: float,
            model: Module,
            features: Tensors,
            target: Tensor
    ) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class EpochPrinter(ArgRepr, EpochCallback):

    def __init__(self, printer: Callable[[str], None] = print) -> None:
        super().__init__(printer)
        self.printer = printer

    def __call__(
            self,
            epoch: int,
            train_loss: int,
            test_loss: int,
            learning_rate: float,
            model: Module,
            features: Tensors,
            target: Tensor
    ) -> None:
        msg = (f'Epoch: {epoch:>4} | learning rate: {learning_rate:8.5f} | '
               f'train loss: {train_loss:8.5f} | test loss: {test_loss:8.5f}')
        return self.printer(msg)

    def close(self) -> None:
        """Does nothing because there is nothing to close."""


class TrainPrinter(ArgRepr):

    def __init__(self, printer: Callable[[str], None] = print) -> None:
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
        if max_epochs_reached:
            self.printer(f'Maximum number of {epoch} epochs exhausted!')
        else:
            msg = (f'Stopping after {epoch} epochs because, even after '
                   f'{epoch - best_epoch} epochs, the loss did not drop '
                   f'below the lowest value of {best_loss:7.5f} seen in epoch '
                   f'{best_epoch}. Recovering checkpoint from that epoch.')
            self.printer(msg)


class Checkpoint(ABC):

    def save(
            self,
            epoch: int,
            loss: float,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> None:
        """docstring"""
        self._save_state(
            epoch,
            loss,
            model.state_dict(),
            {} if optimizer is None else optimizer.state_dict(),
            {} if scheduler is None else scheduler.state_dict()
        )

    def load(
            self,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> tuple[int, float]:
        """docstring"""
        buffer = self._load_state()
        model.load_state_dict(model.state_dict() | buffer['model'])
        if optimizer is not None:
            optimizer.load_state_dict(
                optimizer.state_dict() | buffer['optimizer']
            )
        if scheduler is not None:
            scheduler.load_state_dict(
                scheduler.state_dict() | buffer['scheduler']
            )
        return buffer['epoch'], buffer['loss']

    def reset_parameters(self) -> None:
        self._save_state(0, float('inf'), OrderedDict(), {}, {})

    @abstractmethod
    def _save_state(
            self,
            epoch: int,
            loss: float,
            model: OrderedDict[str, Any],
            optimizer: dict[str, Any],
            scheduler: dict[str, Any]
    ) -> None:
        """docstring"""
        ...

    @abstractmethod
    def _load_state(self) -> State:
        """docstring"""
        ...


class InMemory(Checkpoint):

    def __init__(self) -> None:
        self._buffer = {
            'epoch': 0,
            'loss': float('inf'),
            'model': OrderedDict(),
            'optimizer': {},
            'scheduler': {}
        }

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _save_state(
            self,
            epoch: int,
            loss: float,
            model: OrderedDict[str, Any],
            optimizer: dict[str, Any],
            scheduler: dict[str, Any]
    ) -> None:
        """docstring"""
        self._buffer['epoch'] = epoch
        self._buffer['loss'] = loss
        self._buffer['model'] = copy.deepcopy(model)
        self._buffer['optimizer'] = copy.deepcopy(optimizer)
        self._buffer['scheduler'] = copy.deepcopy(scheduler)

    def _load_state(self) -> State:
        """docstring"""
        return {
            'epoch': self._buffer['epoch'],
            'loss': self._buffer['loss'],
            'model': copy.deepcopy(self._buffer['model']),
            'optimizer': copy.deepcopy(self._buffer['optimizer']),
            'scheduler': copy.deepcopy(self._buffer['scheduler']),
        }

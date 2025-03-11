import warnings
from abc import ABC, abstractmethod
from typing import Any, TypedDict
from collections import OrderedDict
from pathlib import Path
import torch as pt
from ...misc import NotFound, LiteralNotFound
from ..types import Tensor, Module, Optimizer, LRScheduler

__all__ = [
    'State',
    'Checkpoint',
    'InMemory',
    'OnDisk'
]

class State(TypedDict):
    """Container for a snapshot of the state of model training."""
    epoch: int  #: The current epoch.
    loss: float  #: The (train or test) loss at the current epoch.
    model: OrderedDict[str, Any]  #: The state-dict of the model.
    optimizer: dict[str, Any]  #: The state-dict of the optimizer.
    scheduler: dict[str, Any]  #: The state-dict of the scheduler.


class Checkpoint(ABC):
    """Base class to inherit from when implementing custom checkpoints."""

    def __init__(self) -> None:
        self.__counter = 0

    @property
    def counter(self) -> int:
        """How many checkpoints were saved since the last reset."""
        return self.__counter

    def save(
            self,
            epoch: int,
            loss: float,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> None:
        """User-facing method for saving a checkpoint during training.

        Parameters
        ----------
        epoch: int
            The current epoch.
        loss: float
            The (train or test) loss at the current `epoch`.
        model: Module
            The model to checkpoint the state of.
        optimizer: Optimizer, optional
            The optimizer to checkpoint the state of. Defaults to ``None``.
        scheduler: LRScheduler, optional
            The scheduler to checkpoint the state of. Defaults to ``None``.

        """
        self.__counter += 1
        self._save_state({
            'epoch': epoch,
            'loss': loss,
            'model': model.state_dict(),
            'optimizer': {} if optimizer is None else optimizer.state_dict(),
            'scheduler': {} if scheduler is None else scheduler.state_dict()
        })

    def load(
            self,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> tuple[int, float]:
        """User-facing method for loading a checkpoint during training.

        The state retrieved from the checkpoint is merged into the state of
        the objects to update, such that loading a checkpoint before one has
        been saved results in unchanged objects.

        Parameters
        ----------
        model: Module
            The model to load the state for in-place.
        optimizer: Optimizer, optional
            The optimizer to load the state for in-place. Defaults to ``None``.
        scheduler: LRScheduler, optional
            The scheduler to load the state for in-place. Defaults to ``None``.

        Returns
        -------
        epoch: int
            The epoch of the checkpoint.
        loss: float
            The loss from the checkpoint.

        """
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
        """Hard-resets the checkpoint into a pristine, initial state."""
        self.__counter = 0
        self._save_state({
            'epoch': 0,
            'loss': float('inf'),
            'model': OrderedDict(),
            'optimizer': {},
            'scheduler': {}
        })

    @abstractmethod
    def _save_state(self, state: State) -> None:
        """Subclasses must implement how exactly the state is persisted.

        Parameters
        ----------
        state: State
            The state to persist.

        Warnings
        --------
        Make sure the persisted state is decoupled from the actual
        state (e.g., copied) so that it does not change as model
        training continues!

        """
        ...

    @abstractmethod
    def _load_state(self) -> State:
        """Subclasses must implement how exactly the state is retrieved.

        Returns
        -------
        State

        Warnings
        --------
        Make sure that the returned state is decoupled from the persisted
        state (e.g., copied) so that the latter cannot change as model
        training continues!

        """
        ...


class InMemory(Checkpoint):
    """Checkpoint in CPU memory regardless of the device training runs on."""

    def __init__(self) -> None:
        super().__init__()
        self.state = {
            'epoch': 0,
            'loss': float('inf'),
            'model': OrderedDict(),
            'optimizer': {},
            'scheduler': {}
        }

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def _to_cpu(self, obj: Any) -> Any:
        """Recursively copy tensors to CPU and remember original devices."""
        if isinstance(obj, dict):
            clone = {}
            for key, value in obj.items():
                clone[key] = self._to_cpu(value)
            return clone
        if isinstance(obj, list):
            return [self._to_cpu(item) for item in obj]
        if isinstance(obj, Tensor):
            clone = pt.empty_like(obj, device='cpu').copy_(obj, True)
            clone.target = obj.device
            return clone
        return obj

    def _to_device(self, obj: Any) -> Any:
        """Recursively copy backed-up tensors to their original devices."""
        if isinstance(obj, dict):
            clone = {}
            for key, value in obj.items():
                clone[key] = self._to_device(value)
            return clone
        if isinstance(obj, list):
            return [self._to_device(item) for item in obj]
        if isinstance(obj, pt.Tensor):
            return pt.empty_like(obj, device=obj.target).copy_(obj, True)
        return obj

    def _save_state(self, state: State) -> None:
        """Create a deep copy of the combined state in CPU memory."""
        self.state = self._to_cpu(state)

    def _load_state(self) -> State:
        """Create a deep copy of the saved state on the original device(s)."""
        return self._to_device(self.state)


class OnDisk(Checkpoint):
    """Checkpoint the current state of training on disk.

    Parameters
    ----------
    path: str
        Full path to and name of the file used to persist state.
    create: bool, optional
        What to do if the directory where the checkpoint should be saved does
        not exist. Defaults to ``False``.
    not_found: str, optional
        What to do if a checkpoint is loaded from the specified `path`
        but none is there yet. One of "ignore", "warn", or "raise".
        Defaults to "raise". Use the ``NotFound`` enum to avoid typos!

    See Also
    --------
    ~swak.misc.NotFound

    """

    def __init__(
            self,
            path: str,
            create: bool = False,
            not_found: NotFound | LiteralNotFound = NotFound.WARN
    ) -> None:
        super().__init__()
        self.path = str(Path(str(path).strip()).resolve())
        self.create = create
        self.not_found = str(not_found).strip().lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.path}', {self.create})"

    def _save_state(self, state: State) -> None:
        """Save the combined state to file."""
        if self.create:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        pt.save(state, self.path)

    def _load_state(self) -> State:
        """Retrieve the combined state from file."""
        try:
            return pt.load(self.path, weights_only=True)
        except (FileNotFoundError, EOFError) as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'Checkpoint {} not found!\nCreating an empty one.'
                    warnings.warn(msg.format(self.path))
                    self.reset_parameters()
                case NotFound.IGNORE:
                    self.reset_parameters()
                case _:
                    raise error
        return pt.load(self.path, weights_only=True)

import uuid
import fsspec
import torch as pt
from typing import Any
from collections import OrderedDict
from collections.abc import Mapping
from functools import cached_property
from fsspec.spec import AbstractFileSystem
from typing import TypedDict
from ...io import Storage, LiteralStorage
from ...misc import ArgRepr
from ..types import Module, Optimizer, LRScheduler

__all__ = [
    'State',
    'Checkpoint'
]


class State(TypedDict):
    """Container for a snapshot of the state of model training."""
    epoch: int
    loss: float
    model: OrderedDict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


class Checkpoint(ArgRepr):
    """Checkpoint the state of model training on any supported file system.

    On first load, if no checkpoint file exists yet, an empty one is silently
    created and saved so that subsequent loads always succeed.

    Parameters
    ----------
    path: str, optional
        Full absolute path to the checkpoint file.
        Defaults to "/tmp/checkpoint.pt"
    storage: str, optional
        The type of file system to use ("file", "s3", "memory", etc.).
        Defaults to "memory". Use the :class:`Storage` enum to avoid typos.
    chunk_size: int, optional
        Chunk size for file-system IO in MiB. Defaults to 32.
    storage_kws: dict, optional
        Passed as keyword arguments to the fsspec file-system constructor.

    See Also
    --------
    ~swak.io.Storage

    """

    _EMPTY: State = {
        'epoch': 0,
        'loss': float('inf'),
        'model': OrderedDict(),
        'optimizer': {},
        'scheduler': {}
    }

    def __init__(
            self,
            path: str = '/tmp/checkpoint.pt',
            storage: LiteralStorage | Storage = Storage.MEMORY,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = self.__strip(path)
        self.storage = str(Storage(storage))
        self.chunk_size = self.__valid(chunk_size)
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        self.__counter = 0
        super().__init__(
            self.path,
            self.storage,
            self.chunk_size,
            self.storage_kws
        )

    @staticmethod
    def __strip(path: Any) -> str:
        """Try to normalize the path."""
        try:
            stripped = '/' + path.strip(' /')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            msg = 'Path must be a string, not {}!'
            raise TypeError(msg.format(cls)) from error
        if stripped.count('/') < 2:
            msg = f'Path "{stripped}" must not point to the root directory!'
            raise ValueError(msg)
        return stripped

    @staticmethod
    def __valid(chunk_size: Any) -> float:
        """Try to convert chunk_size to a meaningful float."""
        try:
            as_float = float(chunk_size)
        except (TypeError, ValueError) as error:
            cls = type(chunk_size).__name__
            tmp = '"{}" must at least be convertible to a float, unlike {}!'
            msg = tmp.format('chunk_size', cls)
            raise TypeError(msg) from error
        if as_float < 1.0:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('chunk_size', as_float)
            raise ValueError(msg)
        return as_float

    @property
    def counter(self) -> int:
        """How many checkpoints were saved since the last reset."""
        return self.__counter

    @property
    def chunk_bytes(self) -> int:
        """Bytes to read/write in one go."""
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    @cached_property
    def fs(self) -> AbstractFileSystem:
        """Fresh fsspec file system on first use, same thereafter."""
        return fsspec.filesystem(self.storage, **self.storage_kws)

    def save(
            self,
            epoch: int,
            loss: float,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> None:
        """Save a checkpoint of the current training state.

        Parameters
        ----------
        epoch: int
            The current epoch.
        loss: float
            The (train or test) loss at the current `epoch`.
        model: Module
            The model to checkpoint.
        optimizer: Optimizer, optional
            The optimizer to checkpoint. Defaults to ``None``.
        scheduler: LRScheduler, optional
            The scheduler to checkpoint. Defaults to ``None``.

        """
        self._save_state({
            'epoch': epoch,
            'loss': loss,
            'model': model.state_dict(),
            'optimizer': {} if optimizer is None else optimizer.state_dict(),
            'scheduler': {} if scheduler is None else scheduler.state_dict()
        })
        self.__counter += 1

    def reset_parameters(self) -> None:
        """Reset the checkpoint to a pristine initial state."""
        self.__counter = 0
        self._save_state(dict(self._EMPTY))

    def _save_state(self, state: State) -> None:
        """Atomically persist state via a temp file."""
        tmp = f'{self.path}.tmp.{uuid.uuid4().hex}'
        try:
            with self.fs.open(tmp, 'wb', self.chunk_bytes) as file:
                pt.save(state, file)
            self.fs.move(tmp, self.path)
        except Exception:
            self.fs.rm(tmp)
            raise

    def load(
            self,
            model: Module,
            optimizer: Optimizer | None = None,
            scheduler: LRScheduler | None = None
    ) -> tuple[int, float]:
        """Load a checkpoint into the provided training objects in-place.

        If no checkpoint exists yet, an empty one is silently created, leaving
        all objects unchanged.

        Parameters
        ----------
        model: Module
            The model to restore state into.
        optimizer: Optimizer, optional
            The optimizer to restore state into. Defaults to ``None``.
        scheduler: LRScheduler, optional
            The scheduler to restore state into. Defaults to ``None``.

        Returns
        -------
        epoch: int
            The epoch stored in the checkpoint (0 if empty).
        loss: float
            The loss stored in the checkpoint (``inf`` if empty).

        """
        try:
            with self.fs.open(self.path, 'rb', self.chunk_bytes) as file:
                state = pt.load(file, weights_only=True)
        except FileNotFoundError:
            self.reset_parameters()
            state = dict(self._EMPTY)
        model.load_state_dict(model.state_dict() | state['model'])
        if optimizer is not None:
            optimizer.load_state_dict(
                optimizer.state_dict() | state['optimizer']
            )
        if scheduler is not None:
            scheduler.load_state_dict(
                scheduler.state_dict() | state['scheduler']
            )
        return state['epoch'], state['loss']

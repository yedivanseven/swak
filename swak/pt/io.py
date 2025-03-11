"""Save and load (the state of) PyTorch models to and from disk."""

import warnings
import torch as pt
from pathlib import Path
from ..misc import ArgRepr, NotFound, LiteralNotFound
from .types import Module, Device

__all__ = [
    'StateSaver',
    'StateLoader',
    'ModelSaver',
    'ModelLoader'
]


class StateSaver(ArgRepr):
    """Save the state of a model to file.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save a model's ``state_dict()`` to.
        May include any number of string placeholders (i.e., pairs of curly
        brackets) that will be interpolated when instances are called.
        Defaults to the current working directory of the python interpreter.
    create: bool, optional
        What to do if the directory where the state should be saved does
        not exist. Defaults to ``False``.

    """

    def __init__(self, path: str = '', create: bool = False) -> None:
        self.path = str(path).strip()
        self.create = create
        super().__init__(self.path, create)

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        """Save the state of a model, optimizer, or scheduler to file.

        Parameters
        ----------
        model: Module
            Model to save the state of.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        file = str(path.resolve())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        pt.save(model.state_dict(), file)
        return ()


class StateLoader(ArgRepr):
    """Load the state of a model from file.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to the  model's ``state_dict()`` on disk.
        May include any number of string placeholders (i.e., pairs of curly
        brackets) that will be interpolated when instances are called.
        Defaults to the current working directory of the python interpreter.
    map_location: str or Device, optional
        The device to load the state onto. Defaults to ``None`` which loads
        to the PyTorch device(s) that were saved with the model.
    merge: bool, optional
        Whether the loaded state should be merged into the state of the model
        (``True``) or replace its state (``False``). Defaults to ``True``.
    not_found: str, optional
        What to do if the specified file is not found. One of "ignore",
        "warn", or "raise" (use the ``NotFound`` enum to avoid typos).
        Defaults to "raise". If set to "ignore" or "warn" and the specified
        file is not found, `merge` is overridden to ``True``, thus returning
        the unaltered model.

    See Also
    --------
    ~swak.misc.NotFound

    """

    def __init__(
            self,
            path: str = '',
            map_location: Device | str | None = None,
            merge: bool = True,
            not_found: NotFound | LiteralNotFound = NotFound.RAISE
    ) -> None:
        self.path = str(path).strip()
        self.map_location = map_location
        self.merge = merge
        self.not_found = not_found.strip().lower()
        super().__init__(self.path, map_location, merge, self.not_found)

    def __call__(self, model: Module, *parts: str) -> Module:
        """Load the state of a model from file.

        Parameters
        ----------
        model: Module
            Model to load the state of.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        Module
            The `model` with its state restored.

        """
        path = Path(self.path.format(*parts).strip())
        file = str(path.resolve())
        try:
            loaded = pt.load(file, self.map_location, weights_only=True)
            we_should_merge = self.merge
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty state dict.'
                    warnings.warn(msg.format(file))
                    loaded = {}
                    we_should_merge = True
                case NotFound.IGNORE:
                    loaded = {}
                    we_should_merge = True
                case _:
                    raise error
        state = model.state_dict() | loaded if we_should_merge else loaded
        _ = model.load_state_dict(state)
        return model.to(self.map_location) if hasattr(model, 'to') else model


class ModelSaver(ArgRepr):
    """Save an entire model to file.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save the model to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    create: bool, optional
        What to do if the directory where the model should be saved does
        not exist. Defaults to ``False``.

    """

    def __init__(self, path: str = '', create: bool = False) -> None:
        self.path = str(path).strip()
        self.create = create
        super().__init__(self.path, create)

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        """Save a model to file.

        Parameters
        ----------
        model: Module
            The model to save.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        file = str(path.resolve())
        pt.save(model, file)
        return ()


class ModelLoader(ArgRepr):
    """Load a previously saved model from file.

    Parameters
    ----------
    path: str, optional
        Full or partial path to the model to load. If not fully specified here,
        it can be completed on calling the instance. Defaults to the current
        working directory of the python interpreter.
    map_location: str or device, optional
        The device to load the modelo onto. Defaults to ``None`` which loads
        to the PyTorch default device.

    """

    def __init__(
            self,
            path: str = '',
            map_location: Device | str | None = None,
    ) -> None:
        self.path = str(path).strip()
        self.map_location = map_location
        super().__init__(self.path, self.map_location)

    def __call__(self, path: str = '') -> Module:
        """Load a previously saved model from file.

        Parameters
        ----------
        path: str, optional
            Path (including file name) to the model file to load. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        Module
            The loaded model.

        """
        path = str((Path(self.path) / str(path).strip()).resolve())
        obj = pt.load(path, self.map_location, weights_only=False)
        return obj.to(self.map_location) if hasattr(obj, 'to') else obj

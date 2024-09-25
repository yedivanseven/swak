"""Save and load (the state of) PyTorch models to and from disk."""

import warnings
import torch as pt
from pathlib import Path
from ..misc import ArgRepr
from ..text import NotFound
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
    path: str
        Path (including file name) to save a model's ``state_dict()`` to.
        May include any number of string placeholders (i.e., pairs of curly
        brackets) that will be interpolated when instances are called.

    """

    def __init__(self, path: str) -> None:
        self.path = str(Path(path.strip()).resolve())
        super().__init__(self.path)

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
        file = self.path.format(*parts).strip()
        pt.save(model.state_dict(), file)
        return ()


class StateLoader(ArgRepr):
    """Load the state of a model from file.

    Parameters
    ----------
    path: str
        Path (including file name) to the  model's ``state_dict()`` on disk.
        May include any number of string placeholders (i.e., pairs of curly
        brackets) that will be interpolated when instances are called.
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
    NotFound

    """

    def __init__(
            self,
            path: str,
            map_location: Device | str | None = None,
            merge: bool = True,
            not_found: str = NotFound.RAISE
    ) -> None:
        self.path = str(Path(path.strip()).resolve())
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
        file = self.path.format(*parts).strip()
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
    path: str
        Path (including file name) to save the model to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called.

    """

    def __init__(self, path: str) -> None:
        self.path = str(Path(path.strip()).resolve())
        super().__init__(self.path)

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
        file = self.path.format(*parts).strip()
        pt.save(model, file)
        return ()


class ModelLoader(ArgRepr):
    """Load a previously saved model from file.

    Parameters
    ----------
    path: str, optional
        Full or partial path to the file to load. Will be interpreted as a
        directory and prepended as such if the `path` is completed when
        calling instances. Defaults to the current working directory of
        the python interpreter.
    map_location: str or device, optional
        The device to load the modelo onto. Defaults to ``None`` which loads
        to the PyTorch default device.

    """

    def __init__(
            self,
            path: str = '',
            map_location: Device | str | None = None,
    ) -> None:
        self.path = str(Path(path.strip()).resolve())
        self.map_location = map_location
        super().__init__(self.path, self.map_location)

    def __call__(self, path: str = '') -> Module:
        """Load a previously saved model from file.

        Parameters
        ----------
        path: str, optional
            Path (including file name) relative to the `path` specified at
            instantiation. Defaults to an empty string, which results in an
            unchanged `path` on concatenation.

        Returns
        -------
        Module
            The loaded model.

        """
        path = '/' + path.strip(' /') if path.strip(' /') else ''
        model = pt.load(self.path + path, self.map_location, weights_only=True)
        return model.to(self.map_location) if hasattr(model, 'to') else model

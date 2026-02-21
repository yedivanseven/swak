"""Save and load (the state of) PyTorch models to and from any filesystem."""

import warnings
import torch as pt
from typing import Any
from collections.abc import Mapping
from ..io import (
    Mode,
    Writer,
    Reader,
    NotFound,
    LiteralNotFound,
    Storage,
    LiteralStorage
)
from .types import Module, Device

__all__ = [
    'StateSaver',
    'StateLoader',
    'ModelSaver',
    'ModelLoader'
]


class StateSaver(Writer):
    """Save the state of a model to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the file to save a model's ``state_dict()`` to.
        May include two or more forward slashes (subdirectories will be
        created) and string placeholders (i.e., pairs of curly brackets)
        that will be interpolated when instances are called.
    storage: str, optional
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: float, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not a float, or if
        `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    ~swak.io.Storage

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WB,
            chunk_size,
            storage_kws
        )

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        """Save the state of a model, optimizer, or scheduler to a file.

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

        Raises
        ------
        TypeError
            If the `model` does no have a callable `state_dict` method.
        IndexError
            If the `path` given at instantiation has more string placeholders
            that there are `parts`.
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.pt")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        has_state = hasattr(model, 'state_dict') and callable(model.state_dict)
        if not has_state:
            cls = type(model).__name__
            tmp = '"{}" object does not have a callable "state_dict" method!'
            msg = tmp.format(cls)
            raise TypeError(msg)
        if uri := self._uri_from(*parts):
            with self._managed(uri) as file:
                pt.save(model.state_dict(), file)
        return ()


class StateLoader(Reader):
    """Load the state of a model from any supported file system.

    Parameters
    ----------
    path: str
        Full path to the file that holds the model's ``state_dict()``.
        May include two or more forward slashes (subdirectories will be
        created) and string placeholders (i.e., pairs of curly brackets)
        that will be interpolated when instances are called.
    storage: str, optional
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    map_location: str or Device, optional
        The device to load the state onto. Defaults to ``None`` which loads
        to the PyTorch device(s) that were saved with the model.
    merge: bool, optional
        Whether the loaded state should be merged into the state of the model
        (``True``) or replace its state (``False``). This allows loading only
        a partial state. Defaults to ``True``.
    not_found: str, optional
        What to do if the specified file is not found. One of "ignore",
        "warn", or "raise" (use the :class:`NotFound` enum to avoid typos).
        Defaults to "raise". If set to "ignore" or "warn" and the specified
        file is not found, `merge` is overridden to ``True``, thus returning
        the unaltered model.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `parquet_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, `not_found`
        not a permitted string, if the `chunk_size` is smaller than 1 (MiB),
        or if `storage_kws` is not a dictionary.

    See Also
    --------
    ~swak.io.Storage
    ~swak.io.NotFound

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            map_location: Device | str | None = None,
            merge: bool = True,
            not_found: NotFound | LiteralNotFound = NotFound.RAISE
    ) -> None:
        self.map_location = map_location
        self.merge = bool(merge)
        self.not_found = str(NotFound(not_found))
        super().__init__(
            path,
            storage,
            Mode.RB,
            chunk_size,
            storage_kws,
            map_location,
            self.merge,
            self.not_found
        )

    def __call__(self, model: Module, *parts: str) -> Module:
        """Load the state of a model from file on any supported filesystem.

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

        Raises
        ------
        ValueError
            If the final path is directly under root (e.g., "/file.pt")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.
        RuntimeError
            If `merge` set to ``False`` and the loaded state has fewer keys
            than the state of the `model` or if the loaded state has more keys
            than the model.

        """
        uri = self._non_root(self.path.format(*parts))
        try:
            with self._managed(uri) as file:
                loaded = pt.load(file, self.map_location, weights_only=True)
                we_should_merge = self.merge
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File {} not found!\nReturning empty state dict.'
                    warnings.warn(msg.format(uri))
                    loaded = {}
                    we_should_merge = True
                case NotFound.IGNORE:
                    loaded = {}
                    we_should_merge = True
                case _:
                    raise error
        state = model.state_dict() | loaded if we_should_merge else loaded
        # To allow adding keys to the state dict, add `strict` to class API!
        _ = model.load_state_dict(state, strict=True)
        return model.to(self.map_location) if hasattr(model, 'to') else model


class ModelSaver(Writer):
    """Save an entire model to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the file to save the model to. May include two
        or more forward slashes (subdirectories will be created) and string
        placeholders (i.e., pairs of curly brackets) that will be interpolated
        when instances are called.
    storage: str, optional
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: float, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not a float, or if
        `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    ~swak.io.Storage

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WB,
            chunk_size,
            storage_kws
        )

    def __call__(self, model: Module, *parts: str) -> tuple[()]:
        """Save a model to file on any supported file system.

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

        Raises
        ------
        IndexError
            If the `path` given at instantiation has more string placeholders
            that there are `parts`.
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.pt")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        if uri := self._uri_from(*parts):
            with self._managed(uri) as file:
                pt.save(model, file)
        return ()


class ModelLoader(Reader):
    """Load a previously saved model from any supported filesystem.

    Parameters
    ----------
    path: str, optional
        Full path to the model to load.  Since it (or part of it) can also be
        provided later, when the callable instance is called, it is optional
        here. Defaults to an empty string.
    storage: str, optional
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    map_location: str or Device, optional
        The device to load the modelo onto. Defaults to ``None`` which loads
        to the PyTorch default device.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not a float, or if
        `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    ~swak.io.Storage

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            map_location: Device | str | None = None,
    ) -> None:
        self.map_location = map_location
        super().__init__(
            path,
            storage,
            Mode.RB,
            chunk_size,
            storage_kws,
            self.map_location
        )

    def __call__(self, path: str = '') -> Module:
        """Load a previously saved model from any supported filesystem.

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

        Raises
        ------
        ValueError
            If the final path is directly under root (e.g., "/file.pt")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._non_root(path)
        with self._managed(uri) as file:
            model = pt.load(file, self.map_location, weights_only=False)
        return model.to(self.map_location) if hasattr(model, 'to') else model

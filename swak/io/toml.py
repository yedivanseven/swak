from typing import Any
from collections.abc import Mapping
import tomllib
import tomli_w
import warnings
from .writer import Writer
from .reader import Reader
from .types import (
    Toml,
    LiteralStorage,
    Storage,
    Mode,
    NotFound,
    LiteralNotFound
)


class TomlWriter(Writer):
    """Save a TOML file to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the file to save the TOML into. May include two
        or more forward slashes (subdirectories will be created) and string
        placeholders (i.e., pairs of curly brackets) that will be interpolated
        when instances are called.
    storage: str
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the `Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: int, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keywords to the constructor of the file system.
    prune: bool, optional
        Whether to silently drop non-string keys and ``None`` values from the
        dictionary-like object to save as TOML. Defaults to ``False```.
    toml_kws: dict, optional
        Passed on as keyword arguments to the :func:`tomli-w.dump` function.
        See the `tomli-w GitHub page <https://github.com/hukkin/tomli-w>`_
        for  options.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `toml_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if either `storage_kws`
        or `toml_kws` are not dictionaries.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            toml_kws: Mapping[str, Any] | None = None,
            prune: bool = False,
    ) -> None:
        self.toml_kws = {} if toml_kws is None else dict(toml_kws)
        self.prune = bool(prune)
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WB,
            chunk_size,
            storage_kws,
            self.toml_kws,
            self.prune
        )

    @staticmethod
    def __stop_recursion_for(obj: Any) -> bool:
        """Criterion to stop the recursion into a dictionary-like object."""
        try:
            _ = [*obj]
        except TypeError:
            iterable = False
        else:
            iterable = True
        return not iterable or isinstance(obj, str)

    def _pruned(self, toml: Any) -> Toml:
        """Recursively drop fields with ``None`` values and/or non-string keys.

        Parameters
        ----------
        toml
            The dictionary-like object to drop fields from.

        Returns
        -------
        dict
            Dictionary without non-string keys and ``None`` values.

        """
        if self.__stop_recursion_for(toml):
            return toml
        return {
            key: self._pruned(toml[key])
            for key in toml
            if isinstance(key, str) and toml[key] is not None
        } if hasattr(toml, 'keys') else [
            self._pruned(item)
            for item in toml
            if item is not None
        ]

    def __call__(self, toml: Toml, *parts: Any) -> tuple[()]:
        """Serialize a dictionary-like object and write it to TOML file.

        Parameters
        ----------
        toml: dict
            The dictionary-like object to save as TOML.
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
            If the final path is directly under root (e.g., "/file.toml")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.
        TypeError
            If the dictionary-like object contains ``None`` values and
            `pruned` is ``False``.

        """
        if uri := self._uri_from(*parts):
            pruned = self._pruned(toml) if self.prune else toml
            with self._managed(uri) as file:
                tomli_w.dump(pruned, file, **self.toml_kws)
        return ()


class TomlReader(Reader):
    """Read a TOML file from any supported file system.

    Parameters
    ----------
    path: str
        Directory under which the TOML file is located or full path to the
        TOML file. If not fully specified here, it can be completed when
        calling instances.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the `Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    toml_kws: dict, optional
        Passed on as keyword arguments to the :func:`toml.load` function.
    not_found: str, optional
        What to do if the specified TOML file is not found. One of "ignore",
        "warn", or "raise". Defaults to "raise". Use the ``NotFound`` enum to
        avoid typos!

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
    Storage
    NotFound

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            toml_kws: Mapping[str, Any] | None = None,
            not_found: LiteralNotFound | NotFound = 'raise'
    ) -> None:
        self.toml_kws = {} if toml_kws is None else dict(toml_kws)
        self.not_found = str(NotFound(not_found))
        super().__init__(
            path,
            storage,
            Mode.RB,
            chunk_size,
            storage_kws,
            self.toml_kws,
            self.not_found
        )

    def __call__(self, path: str = '') -> Toml:
        """Read a specific TOML file from the specified file system.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, an empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) to the TOML file to read. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        dict
            The parsed contents of the TOML file.

        """
        uri = self._non_root(path)
        try:
            with self._managed(uri) as file:
                toml = tomllib.load(file, **self.toml_kws)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File "{}" not found!\nReturning empty TOML.'
                    warnings.warn(msg.format(uri))
                    toml = {}
                case NotFound.IGNORE:
                    toml = {}
                case _:
                    raise error
        return toml

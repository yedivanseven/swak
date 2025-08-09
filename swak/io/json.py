import json
import warnings
from typing import Any
from collections.abc import Mapping
from pathlib import PurePosixPath
from .writer import Writer
from .reader import Reader
from .types import (
    LiteralStorage,
    Yaml,
    Storage,
    Mode,
    Compression,
    NotFound,
    LiteralNotFound
)


class JsonWriter(Writer):
    """Save a dictionary to a JSON file on any of the supported file systems.

    Parameters
    ----------
    path: str
        The absolute path to the JSON file to save the dictionary into.
        May include two or more forward slashes (subdirectories will be
        created) and string placeholders (i.e., pairs of curly brackets)
        that will be interpolated when instances are called.
    storage: str
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
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
    json_kws: dict, optional
        Passed on as keyword arguments to the :func:`dump` function of
        python's own :mod:`json` module. See the `json documentation
        <https://docs.python.org/3/library/json.html>`_ for options.
    gzip: bool, optional
        Write the JSON to a gzip-compressed JSON file if ``True`` and a plain
        text file if ``False``. If left at ``None``, which is the default,
        file names ending in ".gz" will trigger compression whereas all other
        extensions (if any) will not.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `json_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if either `storage_kws`
        or `json_kws` are not dictionaries.

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
            json_kws: Mapping[str, Any] | None = None,
            gzip: bool | None = None
    ) -> None:
        self.json_kws = {} if json_kws is None else dict(json_kws)
        self.gzip = None if gzip is None else bool(gzip)
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WT,
            chunk_size,
            storage_kws,
            self.json_kws,
            self.gzip
        )

    def __call__(self, obj: Yaml, *parts: Any) -> tuple[()]:
        """Write a dictionary-like object to JSON file the given file system.

        Parameters
        ----------
        obj: dict or list
            The mapping to save as JSON.
        *parts: str
            Fragments that will be interpolated into the `path` given at
            instantiation. Obviously, there must be at least as many as
            there are placeholders in the `path`.

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
            If the final path is directly under root (e.g., "/file.json")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        if uri := self._uri_from(*parts):
            suffix = PurePosixPath(uri).suffix
            zipped = suffix == '.gz' if self.gzip is None else self.gzip
            compression = Compression.GZIP if zipped else None
            with self._managed(uri, compression) as file:
                json.dump(obj, file, **self.json_kws)
        return ()


class JsonReader(Reader):
    """Read a potentially compressed JSON file from any supported file system.

    Parameters
    ----------
    path: str
        Directory under which the JSON file is located or full path to the
        JSON file. If not fully specified here, it can be completed when
        calling instances.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    json_kws: dict, optional
        Passed on as keyword arguments to the :func:`load` function of
        python's own :mod:`json` module. See the `json documentation
        <https://docs.python.org/3/library/json.html>`_ for options.
    not_found: str, optional
        What to do if the specified JSON file is not found. One of "ignore",
        "warn", or "raise". Defaults to "raise". Use the :class:`NotFound`
        enum to avoid typos!
    gzip: bool, optional
        Read the JSON from a gzip-compressed file if ``True`` and a plain
        text file if ``False``. If left at ``None``, which is the default,
        file names ending in ".gz" will be assumed to be compressed whereas
        all other extensions will not.

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
    ~swak.misc.NotFound

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            json_kws: Mapping[str, Any] | None = None,
            not_found: LiteralNotFound | NotFound = 'raise',
            gzip: bool | None = None
    ) -> None:
        self.json_kws = {} if json_kws is None else dict(json_kws)
        self.not_found = str(NotFound(not_found))
        self.gzip = None if gzip is None else bool(gzip)
        super().__init__(
            path,
            storage,
            Mode.RT,
            chunk_size,
            storage_kws,
            self.json_kws,
            self.not_found,
            self.gzip
        )

    def __call__(self, path: str = '') -> Yaml:
        """Read a specific JSON file from the specified file system.

        If `not_found` is set to "warn" or "ignore" and the file cannot be
        found, an empty dictionary is returned.

        Parameters
        ----------
        path: str
            Path (including file name) to the JSON file to read. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        dict
            The parsed contents of the JSON file.

        """
        uri = self._non_root(path)
        try:
            suffix = PurePosixPath(uri).suffix
            zipped = suffix == '.gz' if self.gzip is None else self.gzip
            compression = Compression.GZIP if zipped else None
            with self._managed(uri, compression) as file:
                obj = json.load(file, **self.json_kws)
        except FileNotFoundError as error:
            match self.not_found:
                case NotFound.WARN:
                    msg = 'File "{}" not found!\nReturning empty JSON.'
                    warnings.warn(msg.format(uri))
                    obj = {}
                case NotFound.IGNORE:
                    obj = {}
                case _:
                    raise error
        return obj

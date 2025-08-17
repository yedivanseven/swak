import fsspec
from collections.abc import Mapping
from typing import Any
from functools import cached_property
from fsspec.spec import AbstractFileSystem
from pathlib import PurePosixPath
from ..misc import ArgRepr
from .types import LiteralStorage, Storage


class Find(ArgRepr):
    """List files by prefix and suffix on any supported filesystem.

    Parameters
    ----------
    path: str, optional
        Directory under which files should be discovered. Since it (or part of
        it) can also be provided later, when the callable instance is called,
        it is optional here. Defaults to an empty string.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    suffix: str, optional
        The suffix to filter file names by. Defaults to an empty string,
        which will allow all and any suffixes.
    max_depth: int, optional
        The maximum depth to descend into subdirectories. Defaults to 1.
        If set to ``None``, all subdirectories will be visited recursively.
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.

    Raises
    ------
    TypeError
        If `path` is not a string, `max_depth` is not an int, or if
        `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system schemes,
        `max_depth` is smaller than 1, or if `storage_kws` is not a dictionary.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralStorage | Storage = Storage.FILE,
            suffix: str = '',
            max_depth: int | None = 1,
            storage_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = '/' + self.__stripped(path, 'path', ' /')
        self.storage = str(Storage(storage))
        self.suffix = self.__stripped(suffix, 'suffix', ' .')
        self.suffix = '.' + self.suffix if self.suffix else self.suffix
        self.max_depth = None if max_depth is None else self.__valid(max_depth)
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.suffix,
            self.max_depth,
            self.storage_kws
        )

    # Filesystems that cannot handle leading slashes in paths
    _STRIP_STORAGES = (Storage.GCS,)

    @property
    def strip(self) -> bool:
        """Strip leading slash from path? Necessary on some fle systems."""
        return self.storage in self._STRIP_STORAGES

    @cached_property
    def fs(self) -> AbstractFileSystem:
        """Fresh fsspec file system on first use, same thereafter."""
        return fsspec.filesystem(self.storage, **self.storage_kws)

    @property
    def prefix(self) -> str:
        """File-system specific URI prefix."""
        return f'{self.storage}:/'

    @staticmethod
    def __stripped(obj: Any, name: str, chars: str) -> str:
        """Try to normalize a string argument."""
        try:
            stripped = obj.strip(chars)
        except (AttributeError, TypeError) as error:
            cls = type(obj).__name__
            msg = '"{}" must be a string, not {}!'
            raise TypeError(msg.format(name, cls)) from error
        return stripped

    @staticmethod
    def __valid(max_depth: Any) -> float:
        """Try to convert max_depth to a meaningful int."""
        try:
            as_int = int(max_depth)
        except (TypeError, ValueError) as error:
            cls = type(max_depth).__name__
            tmp = '"{}" must at least be convertible to a int, unlike {}!'
            msg = tmp.format('max_depth', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('max_depth', as_int)
            raise ValueError(msg)
        return as_int

    def _non_root(self, path: str = '') -> str:
        """Append/replace the path given at instantiation on instance call."""
        uri = str(PurePosixPath(self.path) / str(path).strip(' ').rstrip(' /'))
        if uri == '/':
            msg = 'Path must not point to the root directory ("/")!'
            raise ValueError(msg)
        return uri.lstrip('/') if self.strip else uri

    def __call__(self, path: str = '') -> list[str]:
        """List files matching the given criteria on any supported filesystem.

        Parameters
        ----------
        path: str
            Directory under which files should be discovered. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        list
            The full paths to all files under the specified directory,
            filtered for their suffix (if any was given).

        Raises
        ------
        ValueError
            If the final path is directly under root (e.g., "/file.suffix")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._non_root(path)
        files = self.fs.find(
            uri,
            maxdepth=self.max_depth,
            withdirs=False,
            detail=False
        )
        return [
            file.removeprefix(self.prefix)
            for file in files
            if file.endswith(self.suffix)
        ]

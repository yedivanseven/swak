import fsspec
from collections.abc import Mapping
from typing import Any
from functools import cached_property
from pathlib import PurePosixPath
from fsspec.spec import AbstractFileSystem
from ...io.types import LiteralStorage, Storage
from ..misc import ArgRepr


# ToDo. Split into LazyReader and LazyWriter Base classes!
class LazyBase(ArgRepr):
    """Base class for lazily reading from and writing to parquet files.

    Parameters
    ----------
    path: str
        Absolute path to the parquet file (or base directory). May include
        string placeholders (i.e., pairs of curly brackets) that will be
        interpolated by subclasses when instances are called.
    storage: str, optional
        The type of file system to connect to ("file", "s3", "gcs", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    storage_kws: dict, optional
        Passed on as keyword arguments both to the fsspec filesystem
        constructor and as `storage_options` to Polars' scan/sink methods.

    Raises
    ------
    TypeError
        If `path` is not a string or `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system schemes.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            storage_kws: Mapping[str, Any] | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.path = self._strip(path)
        self.storage = str(Storage(storage))
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.storage_kws,
            *args,
            **kwargs
        )

    @cached_property
    def fs(self) -> AbstractFileSystem:
        """Fresh fsspec file system on first use, same thereafter."""
        return fsspec.filesystem(self.storage, **self.storage_kws)

    @staticmethod
    def _strip(path: Any) -> str:
        """Normalize path to an absolute POSIX-style string."""
        try:
            stripped = '/' + path.strip(' /')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            raise TypeError(f'Path must be a string, not {cls}!') from error
        return stripped

    def _non_root(self, path: str) -> str:
        """Assemble and validate a URI, raising if it points to root."""
        appended = str(PurePosixPath(self.path) / path.strip().strip('/'))
        if appended.count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(appended))
        return f'{self.storage}:/{appended}'

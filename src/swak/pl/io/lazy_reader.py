from collections.abc import Mapping
from typing import Any
from pathlib import PurePosixPath
from ..misc import ArgRepr
from .types import LiteralLazyStorage, LazyStorage


class LazyReader(ArgRepr):
    """Base class for scanning polars lazy frames from any filesystem.

    Parameters
    ----------
    path: str, optional
        Directory under which the parquet file is located or its full path.
        Since it (or part of it) can also be provided later, when the callable
        instance is called, it is optional here. Defaults to an empty string.
    storage: str, optional
        The type of file system to scan from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`LazyStorage` enum to avoid typos.
    storage_kws: dict, optional
        Passed on as `storage_options` to polars' scan methods.
    *args
        Additional arguments are reflected in the representation of instances
        but do not affect functionality in any way.
    **kwargs
        Additional keyword arguments are reflected in the representation of
        instances but do not affect functionality in any way.

    Raises
    ------
    TypeError
        If `path` is not a string or `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system schemes.

    See Also
    --------
    LazyStorage

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralLazyStorage | LazyStorage = LazyStorage.FILE,
            storage_kws: Mapping[str, Any] | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.path = self.__strip(path)
        self.storage = str(LazyStorage(storage))
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.storage_kws,
            *args,
            **kwargs
        )

    @property
    def prefix(self) -> str:
        """The URI prefix for the selected storage backend."""
        return '' if self.storage == LazyStorage.FILE else f'{self.storage}:/'

    @staticmethod
    def __strip(path: Any) -> str:
        """Normalize path to an absolute POSIX-style string."""
        try:
            stripped = '/' + path.strip().strip(' /')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            raise TypeError(f'Path must be a string, not {cls}!') from error
        return stripped

    def _non_root(self, path: str = '') -> str:
        """Assemble and validate the URI, raising if it points to root."""
        uri = str(PurePosixPath(self.path) / str(path).strip().rstrip(' /'))
        if uri.count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(uri))
        return f'{self.prefix}{uri}'

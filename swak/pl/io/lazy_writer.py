import fsspec
from collections.abc import Mapping
from typing import Any
from functools import cached_property
from pathlib import PurePosixPath
from fsspec.spec import AbstractFileSystem
from ...io.types import LiteralStorage, Storage
from ..misc import ArgRepr


class LazyWriter(ArgRepr):
    """Base class for sinks of polars lazy frames on any filesystem.

    Parameters
    ----------
    path: str
        The absolute path to the file to sink. May contain any number of string
        placeholders (i.e., pairs of curly brackets) that will be interpolated
        when instances are called. Defaults to "{}".
    storage: str, optional
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    storage_kws: dict, optional
        Passed on as keyword arguments both to the fsspec filesystem
        constructor and as `storage_options` to polars' sink methods.
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
    Storage

    """

    def __init__(
            self,
            path: str = '{}',
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            storage_kws: Mapping[str, Any] | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.path = self.__strip(path)
        self.storage = str(Storage(storage))
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.overwrite,
            self.skip,
            self.storage_kws,
            *args,
            **kwargs
        )

    @cached_property
    def fs(self) -> AbstractFileSystem:
        """Fresh fsspec file system on first use, same thereafter."""
        return fsspec.filesystem(self.storage, **self.storage_kws)

    @staticmethod
    def __strip(path: Any) -> str:
        """Normalize path to an absolute POSIX-style string."""
        try:
            stripped = '/' + path.strip().strip(' /')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            raise TypeError(f'Path must be a string, not {cls}!') from error
        return stripped

    def __non_root_from(self, *parts: Any) -> PurePosixPath:
        """Interpolate parts into the path and validate the result."""
        interpolated = self.__strip(self.path.format(*parts))
        if interpolated.count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(interpolated))
        return PurePosixPath(interpolated)

    def _uri_from(self, *parts: Any) -> str:
        """Check skip/overwrite and create parent directories."""
        path = self.__non_root_from(*parts)
        if self.fs.exists(path):
            if self.skip:
                return ''
            if not self.overwrite:
                msg = f'File "{path}" already exists!'
                raise FileExistsError(msg)
        self.fs.makedirs(str(path.parent), exist_ok=True)
        return f'{self.storage}:/{path}'

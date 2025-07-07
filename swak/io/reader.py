import fsspec
from collections.abc import Generator, Mapping
from typing import Any
from functools import cached_property
from contextlib import contextmanager
from fsspec.spec import AbstractFileSystem
from pathlib import PurePosixPath
from ..misc import ArgRepr
from .types import (
    LiteralStorage,
    Storage,
    LiteralMode,
    Mode,
    LiteralCompression,
    Compression
)


class Reader(ArgRepr):
    """Base class for reading objects from files or blobs on any filesystem.

    Parameters
    ----------
    path: str
        Directory under which the file is located or full path to the file. If
        not fully specified here, it can be completed when calling instances.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the `Storage` enum to avoid typos.
    mode: str, optional
        The mode to open the source file/object/blob in.
        Defaults to "rb". Use the `Mode` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    *args
        Additional arguments are reflected in the representation of instances
        but do not affect functionality in any way.
    **kwargs
        Additional keyword arguments are reflected in the representation of
        instances but do not affect functionality in any way.

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
    Mode

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            mode: LiteralMode | Mode = Mode.RB,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.path = self.__stripped(path)
        self.storage = str(Storage(storage))
        self.mode = str(Mode(mode))
        self.chunk_size = self.__valid(chunk_size)
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.mode,
            self.chunk_size,
            self.storage_kws,
            *args,
            **kwargs
        )

    @property
    def chunk_bytes(self) -> int:
        """Bytes to read from the file system in one go."""
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    @cached_property
    def fs(self) -> AbstractFileSystem:
        """Fresh fsspec file system on first use, same thereafter."""
        return fsspec.filesystem(self.storage, **self.storage_kws)

    @staticmethod
    def __stripped(path: Any) -> str:
        """Try to normalize the path."""
        try:
            stripped = '/' + path.strip(' /')
        except (AttributeError, TypeError) as error:
            cls = type(path).__name__
            msg = 'Path must be a string, not {}!'
            raise TypeError(msg.format(cls)) from error
        return stripped

    @staticmethod
    def __valid(chunk_size: Any) -> float:
        """Try to convert chunk_size to a meaningful integer."""
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

    @contextmanager
    def _managed(
            self,
            uri: str,
            compression: LiteralCompression | Compression | None = None
    ) -> Generator[AbstractFileSystem]:
        """Context manager for atomic reads from the given file system."""
        if compression is not None:
            compression = str(Compression(compression))
        with self.fs.open(
                uri,
                self.mode,
                self.chunk_bytes,
                compression=compression
        ) as file:
            yield file

    def _non_root(self, path: str) -> str:
        """Append/replace the path given at instantiation on instance call."""
        uri = str(PurePosixPath(self.path) / str(path).strip().rstrip('/'))
        if uri.count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(uri))
        return uri

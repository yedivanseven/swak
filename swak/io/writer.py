import uuid
import fsspec
from collections.abc import Generator
from enum import StrEnum
from typing import Any, Literal
from functools import cached_property
from contextlib import contextmanager
from fsspec.spec import AbstractFileSystem
from pathlib import PurePosixPath
from ..misc import ArgRepr

type LiteralMode = Literal['wb', 'wt']
type LiteralStorage = Literal['file', 's3', 'gcs', 'memory']


class Storage(StrEnum):
    """Supported file systems for read/write operations."""
    FILE = 'file'
    S3 = 's3'
    GCS = 'gcs'
    MEMORY = 'memory'


class Mode(StrEnum):
    """Modes for opening files."""
    WB = 'wb'
    WT = 'wt'


class Writer(ArgRepr):
    """Save a pandas or polars dataframe to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the file to save. May contain any number of string
        placeholders (i.e.,  pairs of curly brackets) that will be interpolated
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
    mode: str, optional
        The mode to open the target file/object/blob in.
        Defaults to "wb". Use the `Mode` enum to avoid typos.
    chunk_size: int, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keywords to the constructor of the file system.

    Raises
    ------
    TypeError
        If `path` is not a string,`chunk_size`is not an integer, or if
        `storage_kws` is not a dictionary.
    PermissionError
        If `path` points to a file/object/blob directly under root ("/").
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
            overwrite: bool = False,
            skip: bool = False,
            mode: LiteralMode | Mode = Mode.WB,
            chunk_size: int = 32,
            storage_kws: dict[str, Any] | None = None,
    ) -> None:
        self.path = self.__stripped(path)
        self.storage = str(Storage(storage))
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.mode = str(Mode(mode))
        self.chunk_size = self.__valid(chunk_size)
        self.storage_kws = {} if storage_kws is None else dict(storage_kws)
        super().__init__(
            self.path,
            self.storage,
            self.overwrite,
            self.skip,
            self.mode,
            self.chunk_size,
            self.storage_kws
        )

    @property
    def chunk_bytes(self) -> int:
        """Bytes to flush to the file system in one go."""
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
    def __valid(chunk_size: Any) -> int:
        """Try to convert chunk_size to a meaningful integer."""
        try:
            as_int = int(chunk_size)
        except (TypeError, ValueError) as error:
            cls = type(chunk_size).__name__
            tmp = '"{}" must at least be convertible to integer, unlike {}!'
            msg = tmp.format('chunk_size', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('chunk_size', as_int)
            raise ValueError(msg)
        return as_int

    @contextmanager
    def _managed(self, uri: str) -> Generator[AbstractFileSystem]:
        """Context manager for atomic writes with automatic cleanup."""
        tmp = self._tmp(uri)
        try:
            with self.fs.open(tmp, self.mode, self.chunk_bytes) as file:
                yield file
            self.fs.move(tmp, uri)
        except Exception:
            self.fs.rm(tmp)
            raise

    @staticmethod
    def _tmp(uri: str) -> str:
        """Create a random name for a temporary target file."""
        return f'{uri}.tmp.{uuid.uuid4().hex}'

    def __non_root_from(self, *parts: str) -> PurePosixPath:
        """Interpolate parts into the path and validate the result."""
        path = self.__stripped(self.path.format(*parts))
        if path.count('/') < 2:
            msg = 'Path "{}" cannot point to the root directory ("/")!'
            raise PermissionError(msg.format(path))
        return PurePosixPath(path)

    def _uri_from(self, *parts: str) -> str:
        """Check skip/overwrite and create parent directories."""
        uri = self.__non_root_from(*parts)
        if self.fs.exists(uri):
            if self.skip:
                return ''
            if not self.overwrite:
                msg = f'File "{uri}" already exists!'
                raise FileExistsError(msg)
        self.fs.makedirs(self.fs._parent(uri), exist_ok=True)
        return str(uri)

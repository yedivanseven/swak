import uuid
import fsspec
from typing import Any
from collections.abc import Mapping
from functools import cached_property
from fsspec.spec import AbstractFileSystem
from pathlib import PurePosixPath
from ..misc import ArgRepr
from .types import Storage, LiteralStorage


class Copy(ArgRepr):
    """Efficiently copy a file from one location/filesystem to another.

    Parameters
    ----------
    src_base: str, optional
        Base folder or bucket of the file to read or indeed the full, absolute
        path to the file to read. Because it (or part of it) can also be given
        later at call time, it defaults to an empty string here.
    tgt_base: str, optional
        Base folder or bucket of the file to write or indeed the full, absolute
        path to the file to write. Defaults to ``None``, which will be resolved
        to `src_base`.
    src_storage: str, optional
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    tgt_storage: str, optional
        The type of file system to read from ("file", "s3", etc.).
        Defaults to `src_storage` if not set.
        Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: int, optional
        Chunk size to use when streaming the selected file in MiB.
        Defaults to 32 (MiB).
    src_kws: dict, optional
        Passed on as keywords to the constructor of the source file system.
    tgt_kws: dict, optional
        Passed on as keywords to the constructor of the target file system.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `src_kws` or `tgt_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, the `chunk_size` is smaller than 1 (MiB), or if either
        `src_kws` or `tgt_kws` are not dictionaries.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            src_base: str = '',
            tgt_base: str | None = None,
            src_storage: LiteralStorage | Storage = Storage.FILE,
            tgt_storage: LiteralStorage | Storage | None = None,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            src_kws: Mapping[str, Any] | None = None,
            tgt_kws: Mapping[str, Any] | None = None
    ) -> None:
        self.src_base = self.__strip(src_base)
        if tgt_base is None:
            self.tgt_base = self.src_base
        else:
            self.tgt_base = self.__strip(tgt_base)
        self.src_storage = str(Storage(src_storage))
        if tgt_storage is None:
            self.tgt_storage = self.src_storage
        else:
            self.tgt_storage = str(Storage(tgt_storage))
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.chunk_size = self.__valid(chunk_size)
        self.src_kws = {} if src_kws is None else dict(src_kws)
        self.tgt_kws = {} if tgt_kws is None else dict(tgt_kws)
        super().__init__(
            self.src_base,
            self.tgt_base,
            self.src_storage,
            self.tgt_storage,
            self.overwrite,
            self.skip,
            self.chunk_size,
            self.src_kws,
            self.tgt_kws
        )

    @staticmethod
    def __strip(path: Any) -> str:
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
        """Try to convert chunk_size to a meaningful float."""
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

    @staticmethod
    def _non_root(path: str) -> str:
        """Check that path does not point to a file directly under root."""
        path = PurePosixPath(path)
        if str(path).count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(path))
        if any(part == '..' for part in path.parts):
            msg = 'Relative placeholder ".." is not permitted!'
            raise ValueError(msg)
        return str(path)

    def _src_uri_from(self, path: str) -> str:
        """Normalize path and merge with source URI."""
        path = str(path).strip().rstrip(' /')
        if PurePosixPath(self.src_base) in PurePosixPath(path).parents:
            src_uri = str(PurePosixPath(path))
        else:
            src_uri = str(PurePosixPath(self.src_base) / path.lstrip(' /'))
        return self._non_root(src_uri)

    def _tgt_uri_from(self, src_uri: str) -> str:
        """Merge source URI with target base directory or bucket."""
        stripped = src_uri.removeprefix(self.src_base).strip('/')
        tgt_uri = str(PurePosixPath(self.tgt_base) / stripped)
        return self._non_root(tgt_uri)

    @staticmethod
    def _tmp(uri: str) -> str:
        """Create a random name for a temporary target file."""
        return f'{uri}.tmp.{uuid.uuid4().hex}'

    @property
    def chunk_bytes(self) -> int:
        """Bytes to flush from/to the file system in one go."""
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    @cached_property
    def src_fs(self) -> AbstractFileSystem:
        """Fresh fsspec source file system on first use, same thereafter."""
        return fsspec.filesystem(self.src_storage, **self.src_kws)

    @cached_property
    def tgt_fs(self) -> AbstractFileSystem:
        """Fresh fsspec target file system on first use, same thereafter."""
        return fsspec.filesystem(self.tgt_storage, **self.tgt_kws)

    def __call__(self, path: str = '') -> str:
        """Efficiently copy a single file between file systems.

        Parameters
        ----------
        path: str, optional
            Full path or sub-folder relative to `src_base` and `tgt_base`
            of the file to copy.

        Returns
        -------
        str
            Full path to the target file.

        Raises
        ------
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.txt")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        src_uri = self._src_uri_from(path)
        tgt_uri = self._tgt_uri_from(src_uri)

        if self.tgt_fs.exists(tgt_uri):
            if self.skip:
                return tgt_uri
            if not self.overwrite:
                msg = f'File "{tgt_uri}" already exists!'
                raise FileExistsError(msg)

        parent = str(PurePosixPath(tgt_uri).parent)
        self.tgt_fs.makedirs(parent, exist_ok=True)

        tmp_uri = self._tmp(tgt_uri)
        with self.src_fs.open(src_uri, 'rb', self.chunk_bytes) as src:
            try:
                with self.tgt_fs.open(tmp_uri, 'wb', self.chunk_bytes) as tgt:
                    while chunk := src.read(self.chunk_bytes):
                        tgt.write(chunk)
                self.tgt_fs.mv(tmp_uri, tgt_uri)
            except Exception:
                self.tgt_fs.rm(tmp_uri)
                raise

        return tgt_uri

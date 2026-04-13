from collections.abc import Mapping
from typing import Any
import polars as pl
from polars import LazyFrame
from ...io.types import LiteralStorage, Storage
from .lazy_reader import LazyReader


class Parquet2LazyFrame(LazyReader):
    """Lazily scan a parquet file on any supported file system.

    Parameters
    ----------
    path: str, optional
        Base directory or full path to the parquet file. Since part of it can
        also be provided later, when the callable instance is called, it is
        optional here. Defaults to an empty string.
    storage: str, optional
        The type of file system to read from ("file", "s3", "gcs", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    storage_kws: dict, optional
        Passed on as `storage_options` to :func:`polars.scan_parquet`.
    parquet_kws: dict, optional
        Passed on as additional keyword arguments to polar's top-level
        :func:`scan_parquet`. See the `Polars documentation
        <https://docs.pola.rs/api/python/stable/reference/api/
        polars.scan_parquet.html>`_ for available options.

    Raises
    ------
    TypeError
        If `path` is not a string or either `storage_kws` or `parquet_kws`
        are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system schemes.

    See Also
    --------
    Storage

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralStorage | Storage = Storage.FILE,
            storage_kws: Mapping[str, Any] | None = None,
            parquet_kws: Mapping[str, Any] | None = None
    ) -> None:
        self.parquet_kws = {} if parquet_kws is None else dict(parquet_kws)
        super().__init__(path, storage, storage_kws, self.parquet_kws)

    def __call__(self, path: str = '') -> LazyFrame:
        """Lazily scana parquet file on the specified file system.

        Parameters
        ----------
        path: str, optional
            Path (including file name) to the parquet file to scan. If it
            starts with a forward slash, it is interpreted as absolute;
            otherwise, it is joined to the `path` given at instantiation.
            Defaults to an empty string, which leaves the instantiation
            `path` unchanged.

        Returns
        -------
        LazyFrame
            A Polars :class:`LazyFrame` backed by the specified parquet file.
            No data is read until the frame is collected or sinked.

        Raises
        ------
        ValueError
            If the final path points directly to the root directory.

        """
        uri = self._non_root(path)
        return pl.scan_parquet(
            uri,
            storage_options=self.storage_kws,
            **self.parquet_kws
        )

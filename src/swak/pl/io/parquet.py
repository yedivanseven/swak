from collections.abc import Mapping
from typing import Any
import polars as pl
from polars import LazyFrame
from .types import LiteralLazyStorage, LazyStorage
from .lazy_writer import LazyWriter
from .lazy_reader import LazyReader


class LazyFrame2Parquet(LazyWriter):
    """Sink a polars lazy frame to a parquet file on any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the parquet file to write. May include any number
        of string placeholders (i.e., pairs of curly brackets) that will be
        interpolated when the instance is called.
    storage: str, optional
        The type of file system to write to ("file", "s3", "gcs", etc.).
        Defaults to "file". Use the :class:`LazyStorage` enum to avoid typos.
    storage_kws: dict, optional
        Passed as `storage_options` to :meth:`polars.LazyFrame.sink_parquet`.
    **kwargs
        Passed on as additional keyword arguments to
        :meth:`polars.LazyFrame.sink_parquet`. See the `sink documentation
        <https://docs.pola.rs/api/python/stable/reference/api/
        polars.LazyFrame.sink_parquet.html>`_ for available options.

    Raises
    ------
    TypeError
        If `path` is not a string or `storage_kws` is not a dictionary.
    ValueError
        If `storage` is not among the currently supported file-system schemes.

    See Also
    --------
    LazyStorage

    Note
    ----
    ``sink_parquet`` requires a streaming-compatible query plan. Ensure your
    lazy query is compatible before calling. Polars will raise if it is not.

    """

    def __init__(
            self,
            path: str,
            storage: LiteralLazyStorage | LazyStorage = LazyStorage.FILE,
            storage_kws: Mapping[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        super().__init__(path, storage, storage_kws, **kwargs)

    def __call__(self, ldf: LazyFrame, *parts: Any) -> tuple[()]:
        """Sink a polars lazy frame to a parquet file.

        Parameters
        ----------
        ldf: LazyFrame
            The polars lazy frame to sink.
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
        ValueError
            If the final path is directly under root (e.g., "/file.parquet")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._uri_from(*parts)
        ldf.sink_parquet(uri, storage_options=self.storage_kws, **self.kwargs)
        return ()


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
        Defaults to "file". Use the :class:`LazyStorage` enum to avoid typos.
    storage_kws: dict, optional
        Passed on as `storage_options` to :func:`polars.scan_parquet`.
    **kwargs
        Passed on as additional keyword arguments to polar's top-level
        :func:`scan_parquet` function. See the `scan documentation
        <https://docs.pola.rs/api/python/stable/reference/api/
        polars.scan_parquet.html>`_ for available options.

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
            **kwargs: Any
    ) -> None:
        self.kwargs = kwargs
        super().__init__(path, storage, storage_kws, **kwargs)

    def __call__(self, path: str = '') -> LazyFrame:
        """Lazily scan a parquet file on the specified file system.

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
            If the final path is directly under root (e.g., "/file.parquet")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._non_root(path)
        return pl.scan_parquet(
            uri,
            storage_options=self.storage_kws,
            **self.kwargs
        )

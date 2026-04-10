from collections.abc import Mapping
from typing import Any
from pathlib import PurePosixPath
from polars import LazyFrame
from ...io.types import LiteralStorage, Storage
from .lazy_base import LazyBase


class LazyFrame2Parquet(LazyBase):
    """Sink a polars lazyframe to a parquet file on any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the parquet file to write. May include any number
        of string placeholders (i.e., pairs of curly brackets) that will be
        interpolated when the instance is called.
    storage: str, optional
        The type of file system to write to ("file", "s3", "gcs", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    storage_kws: dict, optional
        Passed on as keyword arguments to the fsspec filesystem constructor
        and as ``storage_options`` to :meth:`polars.LazyFrame.sink_parquet`.
    parquet_kws: dict, optional
        Passed on as additional keyword arguments to
        :meth:`polars.LazyFrame.sink_parquet`. See the `Polars documentation
        <https://docs.pola.rs/api/python/stable/reference/api/
        polars.LazyFrame.sink_parquet.html>`_ for available options.

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

    Note
    ----
    ``sink_parquet`` requires a streaming-compatible query plan. Ensure your
    lazy query is compatible before calling; Polars will raise if it is not.

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            storage_kws: Mapping[str, Any] | None = None,
            parquet_kws: Mapping[str, Any] | None = None
    ) -> None:
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.parquet_kws = {} if parquet_kws is None else dict(parquet_kws)
        super().__init__(
            path,
            storage,
            storage_kws,
            self.overwrite,
            self.skip,
            self.parquet_kws
        )

    def _uri_from(self, *parts: Any) -> str:
        """Interpolate parts, validate, handle skip/overwrite, make parents."""
        path = self._strip(self.path.format(*parts))
        if path.count('/') < 2:
            msg = 'Path "{}" must not point to the root directory ("/")!'
            raise ValueError(msg.format(path))
        uri = f'{self.storage}:/{path}'
        bare = path  # fsspec exists/makedirs operate on the bare path
        if self.fs.exists(bare):
            if self.skip:
                return ''
            if not self.overwrite:
                raise FileExistsError(f'File "{uri}" already exists!')
        self.fs.makedirs(str(PurePosixPath(bare).parent), exist_ok=True)
        return uri

    def __call__(self, lf: LazyFrame, *parts: Any) -> tuple[()]:
        """Sink a Polars LazyFrame to a parquet file.

        Parameters
        ----------
        lf: LazyFrame
            The Polars :class:`LazyFrame` to sink.
        *parts: str
            Fragments interpolated into the `path` given at instantiation.
            There must be at least as many as there are placeholders in the
            `path`.

        Returns
        -------
        tuple
            An empty tuple.

        Raises
        ------
        IndexError
            If the `path` has more placeholders than there are `parts`.
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path points directly to the root directory.

        """
        if uri := self._uri_from(*parts):
            lf.sink_parquet(
                uri,
                storage_options=self.storage_kws,
                **self.parquet_kws
            )
        return ()

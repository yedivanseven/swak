from typing import Any
from collections.abc import Mapping, Callable
from io import BytesIO
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
import pandas as pd
import polars as pl
from ..misc import Bears, LiteralBears
from .writer import Writer
from .reader import Reader
from .types import LiteralStorage, Storage, Mode


class DataFrame2Parquet(Writer):
    """Save a pandas or polars dataframe to any supported file system.

    Parameters
    ----------
    path: str
        The absolute path to the parquet file to save the dataframe into.
        May include two or more forward slashes (subdirectories will be
        created) and string placeholders (i.e., pairs of curly brackets)
        that will be interpolated when instances are called.
    storage: str
        The type of file system to write to ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    overwrite: bool, optional
        Whether to silently overwrite the destination file. Defaults to
        ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the target file already exists.
        Defaults to ``False``.
    chunk_size: int, optional
        Chunk size to use when writing to the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keywords to the constructor of the file system.
    parquet_kws: dict, optional
        Passed on as keyword arguments to the dataframe's write method. See
        the documentation for `to_parquet <https://pandas.pydata.org/
        pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html>`_
        and `write_parquet <https://docs.pola.rs/api/python/stable/reference/
        api/polars.DataFrame.write_parquet.html>`_ methods.

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `parquet_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if either `storage_kws`
        or `parquet_kws` are not dictionaries.

    See Also
    --------
    Storage

    Note
    ----
    Make sure you do a :meth:`reset_index` before you save a pandas dataframe!
    Otherwise, you might have unexpected extra columns in the parquet file
    and potentially undesirable (if not unpredictable) behavior when you
    load it again.

    """

    def __init__(
            self,
            path: str,
            storage: LiteralStorage | Storage = Storage.FILE,
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            parquet_kws: Mapping[str, Any] | None = None
    ) -> None:
        self.parquet_kws = {} if parquet_kws is None else dict(parquet_kws)
        super().__init__(
            path,
            storage,
            overwrite,
            skip,
            Mode.WB,
            chunk_size,
            storage_kws,
            self.parquet_kws
        )

    def __call__(self, df: Pandas | Polars, *parts: Any) -> tuple[()]:
        """Write a pandas or polars dataframe to a supported file system.

        Parameters
        ----------
        df: DataFrame
            The pandas or polars dataframe to save.
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
        FileExistsError
            If the destination file already exists, `skip` is ``False`` and
            `overwrite` is also ``False``.
        ValueError
            If the final path is directly under root (e.g., "/file.parquet")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        writer = 'to_parquet' if hasattr(df, 'to_parquet') else 'write_parquet'
        if uri := self._uri_from(*parts):
            with self._managed(uri) as file:
                getattr(df, writer)(file, **self.parquet_kws)
        return ()


class Parquet2DataFrame(Reader):
    """Read a parquet file from anywhere into a pandas or polars dataframe.

    Parameters
    ----------
    path: str
        Directory under which the parquet file is located or full path to the
        parquet file. Since it (or part of it) can also be provided later,
        when the callable instance is called, it is optional here.
        Defaults to an empty string.
    storage: str
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    parquet_kws: dict, optional
        Passed on as keyword arguments to the dataframe's read method. See
        the documentation for `pandas.read_parquet <https://pandas.pydata.org/
        pandas-docs/stable/reference/api/pandas.read_parquet.html>`_
        and `polars.read_parquet <https://docs.pola.rs/api/python/stable/
        reference/api/polars.read_parquet.html>`_ top-level functions.
    bear: str, optional
        Type of dataframe to return. Can be one of "pandas" or "polars". Use
        the :class:`Bears` enum to avoid typos. Defaults to "pandas".

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `parquet_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    Storage
    ~swak.misc.Bears

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            parquet_kws: Mapping[str, Any] | None = None,
            bear: LiteralBears | Bears = Bears.PANDAS
    ) -> None:
        self.parquet_kws = {} if parquet_kws is None else dict(parquet_kws)
        self.bear = str(Bears(bear))
        super().__init__(
            path,
            storage,
            Mode.RB,
            chunk_size,
            storage_kws,
            self.parquet_kws,
            self.bear
        )

    @property
    def read(self) -> Callable[[BytesIO, ...], Pandas | Polars]:
        """Top-level ``read_parquet`` function of either pandas or polars."""
        return {
            Bears.PANDAS: pd.read_parquet,
            Bears.POLARS: pl.read_parquet
        }[self.bear]

    def __call__(self, path: str = '') -> Pandas | Polars:
        """Read a specific parquet file from the specified file system.

        Parameters
        ----------
        path: str
            Path (including file name) to the parquet file to read. If it
            starts with a backslash, it will be interpreted as absolute,
            if not, as relative to the `path` specified at instantiation.
            Defaults to an empty string, which results in an unchanged `path`.

        Returns
        -------
        DataFrame
            Pandas or polars dataframe.

        Raises
        ------
        ValueError
            If the final path is directly under root (e.g., "/file.parquet")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._non_root(path)
        with self._managed(uri) as file:
            df = self.read(file, **self.parquet_kws)
        return df

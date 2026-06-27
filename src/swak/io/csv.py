from typing import Any, IO
from collections.abc import Mapping, Callable
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
import pandas as pd
import polars as pl
from .types import Bears, LiteralBears, LiteralStorage, Storage, Mode
from .reader import Reader


class Csv2DataFrame(Reader):
    """Read a CSV file from anywhere into a pandas or polars dataframe.

    Parameters
    ----------
    path: str, optional
        Directory under which the CSV file is located or full path to the
        CSV file. Since it (or part of it) can also be provided later,
        when the callable instance is called, it is optional here.
        Defaults to an empty string.
    storage: str, optional
        The type of file system to read from ("file", "s3", etc.).
        Defaults to "file". Use the :class:`Storage` enum to avoid typos.
    chunk_size: float, optional
        Chunk size to use when reading from the selected file system in MiB.
        Defaults to 32 (MiB).
    storage_kws: dict, optional
        Passed on as keyword arguments to the constructor of the file system.
    csv_kws: dict, optional
        Passed on as keyword arguments to the dataframe's read method. See
        the documentation for `pandas.read_csv <https://pandas.pydata.org/
        pandas-docs/stable/reference/api/pandas.read_csv.html>`_
        and `polars.read_csv <https://docs.pola.rs/api/python/stable/
        reference/api/polars.read_csv.html>`_ top-level functions.
    bear: str, optional
        Type of dataframe to return. Can be one of "pandas" or "polars". Use
        the :class:`Bears` enum to avoid typos. Defaults to "pandas".

    Raises
    ------
    TypeError
        If `path` is not a string, `chunk_size` is not an integer or either
        `storage_kws` or `csv_kws` are not dictionaries.
    ValueError
        If `storage` is not among the currently supported file-system
        schemes, `mode` not among the supported file-mode options, the
        `chunk_size` is smaller than 1 (MiB), or if `storage_kws` is not
        a dictionary.

    See Also
    --------
    Storage
    Bears

    """

    def __init__(
            self,
            path: str = '',
            storage: LiteralStorage | Storage = Storage.FILE,
            chunk_size: int = 32,
            storage_kws: Mapping[str, Any] | None = None,
            csv_kws: Mapping[str, Any] | None = None,
            bear: LiteralBears | Bears = Bears.PANDAS
    ) -> None:
        self.csv_kws = {} if csv_kws is None else dict(csv_kws)
        self.bear = str(Bears(bear))
        super().__init__(
            path,
            storage,
            Mode.RT,
            chunk_size,
            storage_kws,
            self.csv_kws,
            self.bear
        )

    @property
    def read(self) -> Callable[[str | IO, ...], Pandas | Polars]:
        """Top-level ``read_csv`` function of either pandas or polars."""
        return {
            Bears.PANDAS: pd.read_csv,
            Bears.POLARS: pl.read_csv
        }[self.bear]

    def __call__(self, path: str = '') -> Pandas | Polars:
        """Read a specific CSV file from the specified file system.

        Parameters
        ----------
        path: str
            Path (including file name) to the CSV file to read. If it
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
            If the final path is directly under root (e.g., "/file.csv")
            because, on local file system, this is not where you want to save
            to and, on object storage, the first directory refers to the name
            of an (existing!) bucket.

        """
        uri = self._non_root(path)
        with self._managed(uri) as file:
            df = self.read(file, **self.csv_kws)
        return df

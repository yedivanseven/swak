from typing import Any
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from ..misc import ArgRepr


class ParquetReader(ArgRepr):
    """Light wrapper around the top-level ``read_parquet`` pandas function.

    Parameters
    ----------
    path: str, optional
        Directory under which the parquet files are located or full path to
        the parquet file. If not fully specified here, the path must be
        completed on calling the instance. Defaults to the current working
        directory of the python interpreter.
    **kwargs
        Keyword arguments passed on to the ``read_parquet`` function call.

    """

    def __init__(self, path: str = '', **kwargs: Any) -> None:
        self.path = str(path).strip()
        self.kwargs = kwargs
        super().__init__(self.path, **kwargs)

    def __call__(self, path: str = '') -> DataFrame:
        """Read one or more parquet file(s) into pandas DataFrame.

        Parameters
        ----------
        path: str, optional
            Path (including file name) to the directory where the parquet
            file(s) to read are located. If it starts with a backslash, it will
            be interpreted as absolute, if not, as relative to the `path`
            specified at instantiation. Defaults to an empty string, which
            results in an unchanged `path`.

        Returns
        -------
        DataFrame

        """
        path = Path(self.path) / str(path).strip()
        return pd.read_parquet(path, **self.kwargs).reset_index(drop=True)

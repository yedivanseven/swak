import os
from typing import Any
import pandas as pd
from pandas import DataFrame
from ..magic import ArgRepr


class ParquetReader(ArgRepr):
    """Light wrapper around the top-level ``read_parquet`` pandas function.

    Parameters
    ----------
    base_dir: str, optional
        Base directory of the parquet file(s) to read. May contain any number
        of forward slashes to access nested subdirectories. Defaults to
        the current working directory of the python interpreter.
    **kwargs
        Keyword arguments passed on to the ``read_parquet`` function call.

    """

    def __init__(self, base_dir: str = '', **kwargs: Any) -> None:
        if stripped := base_dir.strip(' /'):
            self.base_dir = '/' + stripped
        else:
            self.base_dir = os.getcwd()
        self.kwargs = kwargs
        super().__init__(self.base_dir, **kwargs)

    def __call__(self, path: str = '') -> DataFrame:
        """Read one or more parquet file(s) into pandas DataFrame.

        Parameters
        ----------
        path: str, optional
            Path to the file or directory under the `base_dir` to read as
            parquet file(s). May contain any number of forward slashes to
            access nested subdirectories. If not given, an attempt will
            be made to read all files from the `base_dir` as parquet file(s).

        Returns
        -------
        DataFrame

        """
        path = path.rstrip().lstrip(' /')
        full_path = os.path.join(self.base_dir, path)
        df = pd.read_parquet(full_path, **self.kwargs)
        df.reset_index(drop=True, inplace=True)
        return df

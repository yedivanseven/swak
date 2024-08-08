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
        of forward slashes to access nested subdirectories, or string
        placeholders (i.e., curly brackets) to interpolate later. Defaults to
        the current working directory of the python interpreter.
    **kwargs
        Keyword arguments passed on to the ``read_parquet`` function call.

    """

    def __init__(self, base_dir: str = '', **kwargs: Any) -> None:
        self.base_dir = '/' + base_dir.strip(' /') if base_dir else os.getcwd()
        self.kwargs = kwargs
        super().__init__(self.base_dir, **kwargs)

    def __call__(self, path: str = '', *args: Any) -> DataFrame:
        """Read one or more parquet file(s) into pandas DataFrame.

        Parameters
        ----------
        path: str, optional
            Path to the file or directory under the `base_dir` to read as
            parquet file(s). May contain any number of forward slashes to access
            nested subdirectories, or string placeholders (i.e., curly brackets)
            to interpolate with `args`. If not given, an attempt will be made
            to read all files from the `base_dir` as parquet file(s).
        *args
            Additional arguments will be interpolated into the joined string
            of `base_dir` and `path` by calling its `format` method. Obviously,
            the number of args must be equal to (or greater than) the total
            number of placeholders in the combined `base_dir` and `path`.

        Returns
        -------
        DataFrame

        """
        path = path.rstrip().lstrip(' /')
        full_path = os.path.join(self.base_dir, path).format(*args)
        df = pd.read_parquet(full_path, **self.kwargs)
        df.reset_index(drop=True, inplace=True)
        return df

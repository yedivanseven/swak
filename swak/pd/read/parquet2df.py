from typing import Any, BinaryIO
import pandas as pd
from pandas import DataFrame
from ...magic import ArgRepr


type Source = str | list[str] | BinaryIO


class LocalParquet2DataFrame(ArgRepr):
    """Read parquet file(s) directly into pandas DataFrame.

    This is a simple partial for the ``read_parquet`` top-level pandas
    function.  The class is initialized with all keyword arguments to be
    passed to the ``read_parquet`` call. The (callable) object is then
    called with a particular source to read from and previously defined
    keyword arguments are used in that call.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed on to the ``read_parquet`` function call.

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, source: Source) -> DataFrame:
        """Read parquet file(s) into pandas DataFrame.

        Parameters
        ----------
        source: str, iterable of str, or binary IO stream.
            Path to parquet file(s) or a file-like object.

        Returns
        -------
        DataFrame

        """
        df = pd.read_parquet(source, **self.kwargs)
        df.reset_index(drop=True, inplace=True)
        return df


def local_parquet_2_dataframe(source: Source, **kwargs: Any) -> DataFrame:
    """Wrapper around the ``read_parquet`` top-level pandas function.

    Parameters
    ----------
    source: str, iterable of str, or binary IO stream.
        Path to parquet file(s) or a file-like object.
    **kwargs
        Keyword arguments passed on to the ``read_parquet`` function call.

    Returns
    -------
    DataFrame

    """
    df = pd.read_parquet(source, **kwargs)
    df.reset_index(drop=True, inplace=True)
    return df

from typing import Any
import pandas_gbq as pq
from pandas import DataFrame
from ...magic import ArgRepr


class GbqQuery2DataFrame(ArgRepr):
    """Send SQL to Google BigQuery and read the result into a pandas DataFrame.

    This is a simple partial to the top-level ``read_gbq`` function of the
    ``pandas_gbq`` package. As such, it is not suitable for downloading large
    amounts of data (see its `documentation <https://googleapis.dev/python/
    pandas-gbq/latest/>`__ for further details). For query a little (meta)data,
    however, it is most convenient.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed on to the ``read_gbq`` function call.

    See Also
    --------
    GbqQuery2GcsParquet
    GcsParquet2DataFrame
    GcsDir2LocalDir

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def __call__(self, query: str, *args: Any) -> DataFrame:
        """Read results of Google BigQuery SQL into pandas DataFrame.

        Parameters
        ----------
        query: str
            SQL query to be submitted to Google BigQuery.
        *args
            Additional arguments will be interpolated into the query.
            Obviously, the number of args must be equal to (or greater than)
            the total number of placeholders in the query.

        Returns
        -------
        DataFrame
            The results of the SQL query.

        """
        df = pq.read_gbq(query.format(*args), **self.kwargs)
        df.reset_index(drop=True, inplace=True)
        return df

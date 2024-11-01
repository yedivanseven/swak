from typing import Any
import pandas_gbq as pq
from pandas import DataFrame
from ...misc import ArgRepr


# ToDo: Add pola-rs support and rename to GbqQuery2Pandas and GbqQuery2Polars!
class GbqQuery2DataFrame(ArgRepr):
    """Partial of the ``read_gbq`` function in the ``pandas_gbq`` package.

    As such, it may not bet suitable for downloading large amounts of data (see
    its `documentation <https://googleapis.dev/python/pandas-gbq/latest/>`__
    for details).

    Parameters
    ----------
    project: str
        The project to bill for the query/retrieval.
    **kwargs
        Additional keyword arguments passed on to the ``read_gbq`` call.

    See Also
    --------
    GbqQuery2GcsParquet
    GcsParquet2DataFrame
    GcsDir2LocalDir

    """

    def __init__(self, project: str, **kwargs: Any) -> None:
        self.project = project.strip().strip(' /.')
        self.kwargs = kwargs
        super().__init__(self.project, **kwargs)

    def __call__(self, query_or_table: str) -> DataFrame:
        """Read Google BigQuery SQL results or table into a pandas DataFrame.

        Parameters
        ----------
        query_or_table: str
            Table name (including dataset id) or SQL query to be
            retrieved from or submitted to Google BigQuery.

        Returns
        -------
        DataFrame
            The contents of the table or the results of the SQL query.

        """
        return pq.read_gbq(
            query_or_table,
            project_id=self.project,
            progress_bar_type=None,
            **self.kwargs
        )

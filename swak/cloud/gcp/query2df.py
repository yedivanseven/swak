from typing import Any
import pandas_gbq as pq
from pandas import DataFrame
from ...magic import ArgRepr


class GbqQuery2DataFrame(ArgRepr):
    """Partial of the ``read_gbq`` function in the``pandas_gbq`` package.

    As such, it may not bet suitable for downloading large amounts of data (see
    its `documentation <https://googleapis.dev/python/pandas-gbq/latest/>`__
    for details).

    Parameters
    ----------
    project: str
        The project to bill for the query/retrieval.
    location: str
        The physical datacenter location to fetch the data from. See the
        Google Cloud Platform `documentation <https://cloud.google.com/
        bigquery/docs/locations>`__ for options.
    **kwargs
        Additional keyword arguments passed on to the ``read_gbq`` call.

    See Also
    --------
    GbqQuery2GcsParquet
    GcsParquet2DataFrame
    GcsDir2LocalDir

    """

    def __init__(
            self,
            project: str,
            location: str,
            **kwargs: Any
    ) -> None:
        self.project = project.strip(' ./')
        self.location = location.strip().lower()
        self.kwargs = kwargs
        super().__init__(self.project, self.location, **kwargs)

    def __call__(self, query_or_table: str, *args: Any) -> DataFrame:
        """Read results of Google BigQuery SQL into pandas DataFrame.

        Parameters
        ----------
        query_or_table: str
            Table name (including dataset) or SQL query to be
            retrieved from or submitted to Google BigQuery.
        *args
            Additional arguments will be interpolated into the query or table
            name. Obviously, the number of args must be equal to (or greater
            than) the total number of placeholders in the query or table name.

        Returns
        -------
        DataFrame
            The results of the SQL query ro the contents of the table.

        """
        return pq.read_gbq(
            query_or_table.format(*args),
            project_id=self.project,
            location=self.location,
            progress_bar_type=None,
            **self.kwargs
        )

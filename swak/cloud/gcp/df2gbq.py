from typing import Any, Literal
from enum import StrEnum
import pandas_gbq as pq
from pandas import DataFrame
from ...magic import ArgRepr


class IfExists(StrEnum):
    """Specify what to do if the BigQuery table to write to already exists."""
    FAIL = 'fail'
    REPLACE = 'replace'
    APPEND = 'append'


class DataFrame2Gbq(ArgRepr):
    """Partial of the ``to_gbq`` function in the``pandas_gbq`` package.

    As such, it may not be suitable for uploading large amounts of data (see
    its `documentation <https://googleapis.dev/python/pandas-gbq/latest/>`__
    for details).

    Parameters
    ----------
    project: str
        The project to bill for the upload.
    location: str
        The physical datacenter location to load the data onto. See the
        Google Cloud Platform `documentation <https://cloud.google.com/
        bigquery/docs/locations>`__ for options.
    table: str
        The full table name to load data into, including the dataset id.
    if_exists: str, optional
        What to do if the destination table already exists. Can be one of
        "fail", "replace" or "append". Defaults to "fail".  Use the
        ``IfExists`` enum to specify explicitly.
    chunksize: int, optional
        Number of dataframe rows to be inserted in each chunk.
        Defaults to ``None``, resulting in the entire dataframe to be
        inserted at once.
    **kwargs
        Additional keyword arguments passed on to the ``to_gbq`` call.

    See Also
    --------
    IfExists

    """

    def __init__(
            self,
            project: str,
            location: str,
            table: str,
            if_exists: Literal['fail', 'replace', 'append'] = 'fail',
            chunksize: int | None = None,
            **kwargs: Any
    ) -> None:
        self.project = project.strip(' ./')
        self.location = location.strip().lower()
        self.table = table.strip(' ./')
        self.if_exists = if_exists.strip().lower()
        self.chunksize = chunksize
        self.kwargs = kwargs
        super().__init__(
            self.project,
            self.location,
            self.table,
            self.if_exists,
            self.chunksize,
            **kwargs
        )

    def __call__(self, df: DataFrame) -> tuple[()]:
        """Write a pandas DataFrame to a Google BigQuery table.

        Parameters
        ----------
        df: DataFrame
            The dataframe to upload to a Google BigQuery table.

        Returns
        -------
        tuple
            An empty tuple.

        """
        pq.to_gbq(
            df.reset_index(drop=True),
            project_id=self.project,
            location=self.location,
            destination_table=self.table,
            if_exists=self.if_exists,
            chunksize=self.chunksize,
            progress_bar=False,
            **self.kwargs
        )
        return ()

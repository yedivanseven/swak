from typing import Any, Literal
from enum import StrEnum
import pandas_gbq as pq
from pandas import DataFrame
from .exceptions import GbqError
from ...misc import ArgRepr

type LiteralIfExists = Literal['fail', 'replace', 'append']


class IfExists(StrEnum):
    """Specify what to do if the BigQuery table to write to already exists."""
    FAIL = 'fail'
    REPLACE = 'replace'
    APPEND = 'append'


# ToDo: Add pola-rs support and rename to Pandas2Gbq and Polars2Gbq!
class DataFrame2Gbq(ArgRepr):
    """Partial of the ``to_gbq`` function in the ``pandas_gbq`` package.

    As such, it may not be suitable for uploading large amounts of data (see
    its `documentation <https://googleapis.dev/python/pandas-gbq/latest/>`__
    for details).

    Parameters
    ----------
    project: str
        The project to bill for the upload.
    dataset: str
        The id of the dataset where the destination `table` resides.
    table: str, optional
        The name of the table to load data into (excluding the `dataset`) or
        the prefix to it. Defaults to an empty string but will be appended by
        the table name (or suffix) given on calling instances.
    location: str, optional
        The physical datacenter location to load the data onto. See the
        Google Cloud Platform `documentation <https://cloud.google.com/
        bigquery/docs/locations>`__ for options. If not given, non-existing
        tables will be created in the default location of the `dataset`.
        Defaults to an empty string.
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
            dataset: str,
            table: str = '',
            location: str = '',
            if_exists: IfExists | LiteralIfExists = 'fail',
            chunksize: int | None = None,
            **kwargs: Any
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.dataset = dataset.strip().strip(' /.')
        self.table = table.strip().strip(' /.')
        self.location = location.strip().lower()
        self.if_exists = if_exists.strip().lower()
        self.chunksize = chunksize
        self.kwargs = kwargs
        super().__init__(
            self.project,
            self.dataset,
            self.table,
            self.location,
            self.if_exists,
            self.chunksize,
            **kwargs
        )

    def __call__(self, df: DataFrame, table: str = '') -> tuple[()]:
        """Write a pandas DataFrame to a Google BigQuery table.

        Parameters
        ----------
        df: DataFrame
            The dataframe to upload to a Google BigQuery table.
        table: str, optional
            The name (or suffix) of the table to load data into (excluding the
            `dataset`), to be appended to the `table` given at instantiation.
            Defaults to an empty string.

        Returns
        -------
        tuple
            An empty tuple.

        Raises
        ------
        GbqError
            If no `table` was given, neither at instantiation nor when called.

        """
        table = self.table + table.strip().strip(' /.')

        if not table:
            raise GbqError('You must provide a table name!')

        pq.to_gbq(
            df.reset_index(drop=True),
            project_id=self.project,
            destination_table=self.dataset + '.' + table,
            location=self.location or None,
            if_exists=self.if_exists,
            chunksize=self.chunksize,
            progress_bar=False,
            **self.kwargs
        )

        return ()

import time
from typing import Any
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars, from_arrow as pl_from_arrow
from google.cloud.bigquery import QueryJobConfig
from ...misc import ArgRepr
from ...io.types import Bears, LiteralBears
from .clients import Gbq
from .exceptions import GbqError


class GbqQuery2DataFrame(ArgRepr):
    """Results of a Google BigQuery SQL query as a pandas or polars dataframe

    Suitable for small to medium result sets that fit comfortably in memory.
    For large result sets, consider exporting to Google Cloud Storage and
    loading files from there instead.

    Parameters
    ----------
    gbq: Gbq
        An instance of a wrapped GBQ client.
    bears: Bears, optional
        Type of dataframe to return. Can be one of "pandas" or "polars". Use
        the :class:`Bears` enum to avoid typos. Defaults to "pandas".
    config: QueryJobConfig | None, optional
        An instance of ``QueryJobConfig`` (see the `documentation
        <https://docs.cloud.google.com/python/docs/reference/bigquery/
        latest/google.cloud.bigquery.job.QueryJobConfig>`_). If ``None``
        (the default), the default config will be used.
    polling_interval: int, optional
        Job completion is going to be checked for every `polling_interval`
        seconds. Defaults to 5 (seconds).

    Raises
    ------
    TypeError
        Iif `polling_interval` cannot be cast to ``float``.
    ValueError
        If `bears` is not "pandas" or "polars" or if `polling_interval` is
        smaller than 1.

    See Also
    --------
    Gbq
    Bears
    GbqQuery2GcsParquet
    GcsParquet2DataFrame

    """

    def __init__(
            self,
            gbq: Gbq,
            bears: Bears | LiteralBears = 'pandas',
            config: QueryJobConfig | None = None,
            polling_interval: int = 5
    ) -> None:
        self.gbq = gbq
        self.bears = str(Bears(bears))
        self.config = config
        self.polling_interval = self.__valid(polling_interval)
        super().__init__(
            self.gbq,
            self.bears,
            self.config,
            self.polling_interval
        )

    @staticmethod
    def __valid(value: Any) -> float:
        """Try to convert polling interval to a meaningful float."""
        try:
            as_float = float(value)
        except (TypeError, ValueError) as error:
            cls = type(value).__name__
            tmp = '"{}" must at least be convertible to a float, unlike {}!'
            msg = tmp.format('polling_interval', cls)
            raise TypeError(msg) from error
        if as_float < 1.0:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('polling_interval', as_float)
            raise ValueError(msg)
        return as_float

    def __call__(self, query: str) -> Pandas | Polars:
        """Read Google BigQuery SQL results or table into a pandas DataFrame.

        Parameters
        ----------
        query: str
            The SQL query to execute against BigQuery.

        Returns
        -------
        DataFrame
            The results of the SQL query in teh requested dataframe type.

        Raises
        ------
        GbqError
            If the query execution failed for some reason.

        """
        client = self.gbq()
        job = client.query(query, self.config)
        while job.running():
            time.sleep(self.polling_interval)
        if error := job.error_result:
            raise GbqError(
                f"\n{error['reason'].upper()}: {error['message']}")
        rows = job.result()
        arrow = rows.to_arrow()
        if self.bears == Bears.PANDAS:
            return arrow.to_pandas()
        else:
            return pl_from_arrow(arrow)

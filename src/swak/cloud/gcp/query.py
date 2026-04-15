import time
from typing import Any
from google.cloud.bigquery import QueryJobConfig
from ...misc import ArgRepr
from .clients import Gbq
from .exceptions import GbqError


class GbqQuery(ArgRepr):
    """Run a SQL query that does not return anything on Google BigQuery.

    Suitable for DDL statements (CREATE, ALTER, DROP) and DML statements
    (INSERT, UPDATE, DELETE) where the result set is not needed.

    Parameters
    ----------
    gbq: Gbq
        An instance of a wrapped GBQ client.
    config: QueryJobConfig | None, optional
        An instance of ``QueryJobConfig`` (see the `QueryJobConfig docs
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
        If `polling_interval` is smaller than 1.

    See Also
    --------
    Gbq

    """

    def __init__(
            self,
            gbq: Gbq,
            config: QueryJobConfig | None = None,
            polling_interval: int = 5
    ) -> None:
        self.gbq = gbq
        self.config = config
        self.polling_interval = self.__valid(polling_interval)
        super().__init__(
            self.gbq,
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

    def __call__(self, query: str) -> tuple[()]:
        """Run a query that does not return anything on Google BigQuery.

        Parameters
        ----------
        query: str
            The SQL query to execute against BigQuery (typically DDL or DML).

        Returns
        -------
        tuple
            If the job finishes without errors, an empty tuple is returned.

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
        return ()

import time
from typing import Any
from google.cloud.bigquery import Client, QueryJobConfig
from ...misc import ArgRepr
from .exceptions import GbqError


class GbqQuery(ArgRepr):
    """Run a SQL query that does not return anything on Google BigQuery.

    Parameters
    ----------
    project: str
        The name of the Google billing project.
    polling_interval: int, optional
        Job completion is going to be checked for every `polling_interval`
        seconds. Defaults to 5 (seconds).
    priority: str, optional
        Priority the query job should be run as. Can be either "BATCH" or
        "INTERACTIVE". Defaults to "BATCH". Use the `QueryPriority
        <https://cloud.google.com/python/docs/reference/bigquery/latest/
        google.cloud.bigquery.job.QueryPriority>`__ enum to avoid typos.
    **kwargs
        Additional keyword arguments are passed to the constructor of the
        Google BigQuery ``Client`` (see `documentation <https://cloud.google.
        com/python/docs/reference/bigquery/latest/google.cloud.bigquery.
        client.Client#parameters>`__ for options).

    Note
    ----
    Typical use cases would be to create, move, alter, or delete tables.
    As such, the possibility to route the output of the query into a
    destination table is not foreseen.

    """

    def __init__(
            self,
            project: str,
            polling_interval: int = 5,
            priority: str = 'BATCH',
            **kwargs: Any
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.polling_interval = polling_interval
        self.priority = priority.strip().upper()
        self.kwargs = kwargs
        super().__init__(
            self.project,
            self.polling_interval,
            self.priority,
            **kwargs
        )

    def __call__(self, query: str, **kwargs: Any) -> tuple[()]:
        """Run a query that does not return anything on Google BigQuery.

        Parameters
        ----------
        query: str
            The SQL query to fire using the pre-configured client.
        **kwargs
            Additional keyword arguments are passed to the constructor of the
            Google BigQuery ``QueryJobConfig``. See `documentation
            <https://cloud.google.com/python/docs/reference/bigquery/latest/
            google.cloud.bigquery.job.QueryJobConfig>`__ for options.

        Returns
        -------
        tuple
            If the job finishes without errors, an empty tuple is returned.

        Raises
        ------
        GbqError
            If the ``QueryJob`` finishes and returns and error.

        """
        client = Client(self.project, **self.kwargs)
        config = QueryJobConfig(priority=self.priority, **kwargs)
        job = client.query(query, config)
        while job.running():
            time.sleep(self.polling_interval)
        if error := job.error_result:
            raise GbqError(f"\n{error['reason'].upper()}: {error['message']}")
        return ()

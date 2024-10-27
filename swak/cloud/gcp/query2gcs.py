import time
import uuid
import os
from typing import Any
from google.cloud import bigquery as gbq
from google.cloud import storage as gcs
from .exceptions import GbqError
from ...misc import ArgRepr


class GbqQuery2GcsParquet(ArgRepr):
    """Export the results of an SQL query to a bucket on Google Cloud Storage.

    The SQL query will be fired against Google BigQuery. In essence, an
    ``EXPORT`` clause with the specified parameters interpolated is inserted
    into the query, after the last semicolon, before everything else.
    As such, no semicolon-separated sub-queries are allowed, but variable
    declaration and setting is fine. Results will be saved as multiple,
    sequentially numbered, `snappy`-compressed parquet files.

    Parameters
    ----------
    project: str
        The name of the Google BigQuery billing project.
    bucket: str
        The Google Cloud Storage bucket to export to. Note that, unless you
        have set up some different "wiring" yourself, the project of the
        bucket is the same as `project`.
    prefix: str, optional
        Prefix of the blob location, where parquet files will reside. If
        none is given here, one can be provided on calling the instance.
        If both are given, they will be concatenated and, if neither
        is given, a UUID will be generated. Defaults to an empty string.
    overwrite: bool, optional
        Blobs with the given bucket/prefix combination may already exist on
        Google Cloud Storage. If ``True`` these are overwritten, else an
        exception is raised. Defaults to ``False``.
    skip: bool, optional
        Blobs with the given bucket/prefix combination may already exist on
        Google Cloud Storage. If that is the case, and `skip` is ``True``,
        nothing will be done at all. Defaults to ``False``
    polling_interval: int, optional
        Job completion is going to be checked for every `polling_interval`
        seconds. Defaults to 5 (seconds).
    priority: str, optional
        Priority the query job should be run as. Can be either "BATCH" or
        "INTERACTIVE". Defaults to "BATCH". Use the `QueryPriority
        <https://cloud.google.com/python/docs/reference/bigquery/latest/
        google.cloud.bigquery.job.QueryPriority>`__ enum to avoid typos.
    gbq_kws: dict, optional
        Additional keyword arguments to passed to the Google BigQuery client.
        Defaults to ``None``, which results in an empty dictionary.
    gcs_kws: dict, optional
        Additional keyword arguments to passed to the Google Storage client.
        Defaults to ``None``, which results in an empty dictionary.

    """

    _template = """    EXPORT DATA OPTIONS(
        uri="gs://{}/{}*.parquet"
      , format="PARQUET"
      , compression="SNAPPY"
      , overwrite={}
    ) AS"""

    def __init__(
            self,
            project: str,
            bucket: str,
            prefix: str = '',
            overwrite: bool = False,
            skip: bool = False,
            polling_interval: int = 5,
            priority: str = 'BATCH',
            gbq_kws: dict[str, Any] | None = None,
            gcs_kws: dict[str, Any] | None = None,
    ) -> None:
        self.project = project.strip().strip(' /.')
        self.bucket = bucket.strip().strip(' /.')
        self.prefix = prefix.strip().strip(' /.')
        self.overwrite = overwrite
        self.skip = skip
        self.polling_interval = polling_interval
        self.priority = priority.strip().upper()
        self.gbq_kws = {} if gbq_kws is None else gbq_kws
        self.gcs_kws = {} if gcs_kws is None else gcs_kws
        super().__init__(
            self.project,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            self.polling_interval,
            self.priority,
            self.gbq_kws,
            self.gcs_kws,
        )

    def __call__(self, query: str, prefix: str = '', **kwargs: Any) -> str:
        """Export the results of a SQL query to Google Cloud Storage.

        Parameters
        ----------
        query: str
            The SQL query to fire using the pre-configured client.
        prefix: str, optional
            Prefix of the blob location, where parquet files will reside. If
            none is given, the prefix specified at instantiation will be used.
            If both are given, they will be concatenated and, if neither
            is given, a UUID will be generated. Defaults to an empty string.
        **kwargs
            Additional keyword arguments are passed to the constructor of the
            Google BigQuery ``QueryJobConfig``. See `documentation
            <https://cloud.google.com/python/docs/reference/bigquery/latest/
            google.cloud.bigquery.job.QueryJobConfig>`__ for options.

        Returns
        -------
        str
            If the query finishes without errors, the given or generated prefix
            is returned, so that blobs with the exported data can be retrieved.

        Raises
        ------
        GbqError
            If the submitted ``QueryJob`` finishes and returns and error.

        """
        scripts, main = self.__split(query)
        prefix, header = self.__render(prefix.strip().strip(' /.'))
        if self.__skip_query_for(prefix):
            return prefix
        client = gbq.Client(self.project, **self.gbq_kws)
        config = gbq.QueryJobConfig(priority=self.priority, **kwargs)
        job = client.query('\n'.join([scripts, header, main]), config)
        while job.running():
            time.sleep(self.polling_interval)
        if error := job.error_result:
            raise GbqError(f"\n{error['reason'].upper()}: {error['message']}")
        return prefix

    def __skip_query_for(self, prefix: str) -> bool:
        """Do blobs with prefix exist on Google Storage? Overwrite them?"""
        client = gcs.Client(self.project, **self.gcs_kws)
        blobs = list(client.list_blobs(self.bucket, prefix=prefix))
        if blobs and self.skip:
            return True
        if blobs and self.overwrite:
            bucket = client.get_bucket(self.bucket)
            bucket.delete_blobs(blobs)
        return False

    def __render(self, prefix: str) -> tuple[str, str]:
        """Render the EXPORT header template with the given parameters."""
        prefix = os.path.join(self.prefix, prefix) if prefix else self.prefix  # noqa: PTH118
        prefix = prefix or str(uuid.uuid4())
        prefix = prefix + '/' if prefix else ''
        header = self._template.format(
            self.bucket,
            prefix,
            str(self.overwrite).lower()
        )
        return prefix, header

    @staticmethod
    def __split(query: str) -> tuple[str, str]:
        """Split query into scripts (before last semicolon) and main part."""
        split = query.split(';')
        separator = ';\n' if len(split) > 1 else ''
        scripts = ';'.join(split[:-1]) + separator
        return scripts, split[-1]

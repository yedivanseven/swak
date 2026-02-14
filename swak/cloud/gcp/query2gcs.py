import time
import uuid
from typing import Any
from google.cloud.bigquery import QueryJobConfig
from google.cloud.storage import Client as GcsClient
from .exceptions import GbqError
from .clients import Gbq
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
    gbq: Gbq
        An instance of a wrapped GBQ client.
    path: str, optional
        The path to the cloud storage "directory", where parquet files will
        reside in "bucket/prefix/" form. May contain any number of string
        placeholders (i.e.,  pairs of curly brackets) that will be interpolated
        when instances are called. If the prefix part is empty after
        interpolation, a randomly generated UUID will be used.
        Defaults to "{}".
    overwrite: bool, optional
        Blobs with the given bucket/prefix combination may already exist on
        Google Cloud Storage. If ``True`` these are overwritten, else an
        exception is raised. Defaults to ``False``.
    skip: bool, optional
        Blobs with the given bucket/prefix combination may already exist on
        Google Cloud Storage. If that is the case, and `skip` is ``True``,
        nothing will be done at all. Defaults to ``False``
    config: QueryJobConfig | None, optional
        An instance of ``QueryJobConfig`` (see the `documentation
        <https://docs.cloud.google.com/python/docs/reference/bigquery/
        latest/google.cloud.bigquery.job.QueryJobConfig>`_). If ``None``
        (the default), the default config will be used.
    polling_interval: int, optional
        Job completion is going to be checked for every `polling_interval`
        seconds. Defaults to 5 (seconds).
    **kwargs:
        Additional keyword arguments are passed to the constructor of the
        Google Cloud Storage (GCS) client, overwriting common options that
        are plucked from the Google BigQuery client.

    Raises
    ------
    TypeError
        If `path` is not a string or `polling_interval` cannot be cast to
        `float`.
    ValueError
        If `path` is empty after sanitation or `polling_interval` is < 1.

    """

    _TEMPLATE = """    EXPORT DATA OPTIONS(
        uri="gs://{}/{}/*.parquet"
      , format="PARQUET"
      , compression="SNAPPY"
      , overwrite={}
    ) AS"""

    def __init__(
            self,
            gbq: Gbq,
            path: str= '{}',
            overwrite: bool = False,
            skip: bool = False,
            config: QueryJobConfig | None = None,
            polling_interval: int = 5,
            **kwargs: Any
    ) -> None:
        self.gbq = gbq
        self.path = self.__strip(path)
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.config = config
        self.polling_interval = self.__valid(polling_interval)
        self.kwargs = kwargs
        super().__init__(
            self.gbq,
            self.path,
            self.overwrite,
            self.skip,
            self.config,
            self.polling_interval,
            **self.kwargs
        )

    @staticmethod
    def __strip(path: str) -> str:
        """Make sure that the path is a non-empty string."""
        try:
            stripped = path.strip().strip(' /.')
        except AttributeError as error:
            cls = type(path).__name__
            tmp = '"{}" must be a string, unlike {}!'
            msg = tmp.format('path', cls)
            raise TypeError(msg) from error
        if not stripped:
            tmp = '"{}" must not be empty after sanitization!'
            msg = tmp.format('path')
            raise ValueError(msg)
        return stripped

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

    @property
    def options(self) -> dict[str, Any]:
        """Options used for the Google Cloud Storage client."""
        return {
            'project': self.gbq.project,
            'credentials': self.gbq.kwargs.get('credentials'),
            '_http': self.gbq.kwargs.get('_http'),
            'client_info': self.gbq.kwargs.get('client_info'),
            'client_options': self.gbq.kwargs.get('client_options')
        } | self.kwargs

    @property
    def flag(self) -> str:
        """Stringified version of the otherwise boolean `overwrite` option."""
        return str(self.overwrite).lower()

    def __call__(self, query: str, *parts: Any) -> str:
        """Export the results of a SQL query to Google Cloud Storage.

        Parameters
        ----------
        query: str
            The SQL query to fire using the pre-configured client.
       *parts: str
            Fragments that will be interpolated into the `path` given at
            instantiation. Obviously, there must be at least as many as
            there are placeholders in the `path`.

        Returns
        -------
        str
            If the query finishes without errors, the given or generated prefix
            is returned, so that blobs with the exported data can be retrieved.

        Raises
        ------
        ValueError
            If `query` is empty of if `path` is empty after inserting `parts`.
        FileExistsError
            If `overwrite` is set to `True` and files with the given prefix
            already exists in the given bucket.
        GbqError
            If the submitted ``QueryJob`` finishes and returns and error.

        """
        bucket, prefix = self._normalize(*parts)
        if self._skip_query_for(bucket, prefix):
            return f'{bucket}/{prefix}/'
        header = self._render(bucket, prefix)
        scripts, main = self._split(query)
        client = self.gbq()
        job = client.query('\n'.join([scripts, header, main]), self.config)
        while job.running():
            time.sleep(self.polling_interval)
        if error := job.error_result:
            raise GbqError(f"\n{error['reason'].upper()}: {error['message']}")
        return f'{bucket}/{prefix}/'

    def _normalize(self, *parts: Any) -> tuple[str, str]:
        """Sanitize interpolated path and split into bucket and prefix."""
        path = self.__strip(self.path.format(*parts))
        parts = path.split('/')
        bucket = parts[0]
        prefix = '/'.join(filter(None, parts[1:])) or str(uuid.uuid4())
        return bucket, prefix

    def _skip_query_for(self, bucket: str, prefix: str) -> bool:
        """Do blobs with prefix exist on Google Storage? Overwrite them?"""
        client = GcsClient(**self.options)
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        if not blobs:
            return False
        if self.skip:
            return True
        if self.overwrite:
            client.get_bucket(bucket).delete_blobs(blobs)
            return False
        tmp = '"{}/{}/" is not empty! Set either skip=True or overwrite=True.'
        msg = tmp.format(bucket, prefix)
        raise FileExistsError(msg)

    def _render(self, bucket: str, prefix: str) -> str:
        """Render the EXPORT header template with the given parameters."""
        return self._TEMPLATE.format(bucket, prefix, self.flag)

    @staticmethod
    def _split(query: str) -> tuple[str, str]:
        """Split query into scripts (before last semicolon) and main part."""
        stripped = query.strip().strip(' ;')
        if not stripped:
            raise ValueError('The query must not be empty!')
        parts = stripped.split(';')
        if len(parts) < 2:
            return '', parts[0]
        scripts = ';\n'.join(parts[:-1]) + ';\n'
        return scripts, parts[-1]

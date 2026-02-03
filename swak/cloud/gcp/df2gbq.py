import time
from typing import Any
from io import BytesIO
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
from google.cloud.bigquery import LoadJobConfig, SourceFormat
from google.cloud.exceptions import NotFound
from .exceptions import GbqError
from .clients import Gbq
from ...misc import ArgRepr


# ToDo: Write unittests
class ParquetLoadJobConfig(ArgRepr):
    """A LoadJobConfig with source format locked to PARQUET.

    All other options, including parquet-specific ones, can be set freely
    via keyword arguments. See `API reference <https://cloud.google.com/python
    /docs/reference/bigquery/latest/google.cloud.bigquery.job.LoadJobConfig>`_
    for options.

    Parameters
    ----------
    **kwargs
        Keyword arguments passed on to ``LoadJobConfig``. The `source_format`
        argument is ignored if given.

    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = (kwargs.pop('source_format', ''), kwargs)[1]
        super().__init__(**self.kwargs)

    def __call__(self, *_: Any, **__: Any) -> LoadJobConfig:
        """New LoadJobConfig on every call, ignoring any arguments."""
        return LoadJobConfig(source_format=SourceFormat.PARQUET, **self.kwargs)


class DataFrame2Gbq(ArgRepr):
    """Upload a pandas or polars dataframe into a Google BigQuery table.

    Not suitable for uploading large amounts of data. Fine

    Parameters
    ----------
    gbq: Gbq
        An instance of a wrapped GBQ client.
    dataset: str
        The id of the dataset where the destination `table` resides.
    table: str, optional
        The name of the table to load data into (excluding the `dataset`) or
        the prefix to it. May contain any number of string placeholders (i.e.,
        pairs of curly brackets) that will be interpolated when instances are
        called. Defaults to "{}".
    location: str, optional
        The physical datacenter location to load the data onto. See the
        Google Cloud Platform `documentation <https://docs.cloud.google.com
        bigquery/docs/locations>`__ for options. If not given, non-existing
        tables will be created in the default location of the `dataset`.
        Defaults to an empty string.
    config: ParquetLoadJobConfig | None, optional
        An instance of the wrapped ``LoadJobConfig`` with ``source_format``
        locked to ``PARQUET``. All other load job options can be set freely
        via that wrapper. If ``None`` (the default), a default config with
        only ``source_format`` set will be used.
    chunk_size: int, optional
        Chunk size (in MiB) to use when uploading. If ``None`` (the default)
        or large, resumable upload will be used. Otherwise, multipart upload
        of chunks with the given size will be used.
    polling_interval: int, optional
        Job completion is going to be checked for every `polling_interval`
        seconds. Defaults to 5 (seconds).
    **kwargs
        Additional keyword arguments passed on to the dataframe method that
        writes to parquet file.

    Raises
    ------
    AttributeError
        If `location` is not a string.
    TypeError
        If `dataset` or `table` are not strings or if `chunk_size` or
        `polling_interval` cannot be cast to ``float``.
    ValueError
        If either `dataset` or `table` are emtpy strings or if either of
        `chunk_size` or `polling_interval` are smaller than 1.

    See Also
    --------
    Gbq
    ParquetLoadJobConfig

    """

    def __init__(
            self,
            gbq: Gbq,
            dataset: str,
            table: str = '{}',
            location: str = 'europe-north1',
            config: ParquetLoadJobConfig | None = None,
            chunk_size: int | None = None,
            polling_interval: int = 5,
            **kwargs: Any
    ) -> None:
        self.gbq = gbq
        self.dataset = self.__strip(dataset, 'dataset')
        self.table = self.__strip(table, 'table')
        self.location = location.strip().lower()
        self.config = config or ParquetLoadJobConfig()
        self.chunk_size = self.__valid(chunk_size, 'chunk_size')
        self.polling_interval = self.__valid(
            polling_interval, 'polling_interval'
        )
        self.kwargs = dict(kwargs)
        super().__init__(
            self.gbq,
            self.dataset,
            self.table,
            self.location,
            self.config,
            self.chunk_size,
            self.polling_interval,
            **self.kwargs
        )

    @staticmethod
    def __strip(value: Any, name: str) -> str:
        """Get the last dot-separated segment and strip surrounding noise."""
        try:
            result = value.split('.')[-1].strip().strip(' ./')
        except AttributeError as error:
            cls = type(value).__name__
            tmp = '"{}" must be a string, unlike {}!'
            msg = tmp.format(name, cls)
            raise TypeError(msg) from error
        if not result:
            tmp = '"{}" must not be empty after sanitization!'
            msg = tmp.format(name)
            raise ValueError(msg)
        return result

    @staticmethod
    def __valid(value: Any, name: str) -> float | None:
        """Try to convert chunk_size/polling interval to meaningful floats."""
        if value is None:
            return value
        try:
            as_float = float(value)
        except (TypeError, ValueError) as error:
            cls = type(value).__name__
            tmp = '"{}" must at least be convertible to a float, unlike {}!'
            msg = tmp.format(name, cls)
            raise TypeError(msg) from error
        if as_float < 1.0:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format(name, as_float)
            raise ValueError(msg)
        return as_float

    @property
    def chunk_bytes(self) -> int | None:
        """Number of bytes to upload per part."""
        if self.chunk_size is None:
            return self.chunk_size
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    def __call__(self, df: Pandas | Polars, *parts: Any) -> tuple[()]:
        """Write a pandas DataFrame to a Google BigQuery table.

        Parameters
        ----------
        df: DataFrame
            The pandas or polars dataframe to upload into a BigQuery table.
        *parts: str
            Fragments that will be interpolated into the `table` given at
            instantiation. Obviously, there must be at least as many as
            there are placeholders in the `table`.

        Returns
        -------
        tuple
            An empty tuple.

        Raises
        ------
        GbqError
            If the upload failed for some reason.

        """
        table = self.__strip(self.table.format(*parts), 'table')
        destination = self.dataset + '.' + table
        writer = 'to_parquet' if hasattr(df, 'to_parquet') else 'write_parquet'
        client = self.gbq()

        try:
            location = client.get_table(destination).location
        except NotFound:
            location = self.location

        with BytesIO() as stream:
            getattr(df, writer)(stream, **self.kwargs)
            job = client.load_table_from_file(
                file_obj=stream,
                destination=destination,
                rewind=True,
                size=self.chunk_bytes,
                location=location,
                job_config=self.config()
            )
            while job.running():
                time.sleep(self.polling_interval)
            if error := job.error_result:
                raise GbqError(
                    f"\n{error['reason'].upper()}: {error['message']}")

        return ()

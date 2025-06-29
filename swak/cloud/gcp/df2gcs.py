from typing import Any
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
from ...misc import ArgRepr
from .exceptions import GcsError
from .clients import Gcs


class DataFrame2GcsParquet(ArgRepr):
    """Upload a pandas or polars dataframe to Google Cloud Storage (GCS).

    Parameters
    ----------
    gcs: Gcs
         An instance of a wrapped GCS client.
    bucket: str
        The name of the bucket to upload to.
    prefix: str, optional
        The prefix of the parquet file to upload the dataframe to. May include
        any number of string placeholders (i.e., pairs of curly brackets) that
        will be interpolated when instances are called.
        Defaults to an empty string.
    overwrite: bool, optional
        Whether to silently overwrite the destination blob on GCS. Defaults
        to ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the destination blob on GCS already
        exists. Defaults to ``False``.
    chunk_size: int, optional
        Chunk size to write from Google Cloud Storage in one API call in MiB.
        Defaults to 40 MiB.
    **kwargs
        Additional keyword arguments are passed on to the ``to_parquet``
        or ``write_parquet`` method of the dataframe.


    Raises
    ------
    AttributeError
        If either `bucket` or `prefix` are not, in fact, strings.

    See Also
    --------
    Gcs

    """

    def __init__(
            self,
            gcs: Gcs,
            bucket: str,
            prefix: str = '',
            overwrite: bool = False,
            skip: bool = False,
            chunk_size: int = 40,
            **kwargs: Any,
    ) -> None:
        self.gcs = gcs
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip(' /')
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.chunk_size = self.__valid(chunk_size)
        self.kwargs = kwargs
        super().__init__(
            self.gcs,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            **self.kwargs
        )

    @property
    def chunk_bytes(self) -> int:
        """Bytes to write to Google Cloud Storage in one API call."""
        in_bytes = self.chunk_size * 1024 * 1024
        in_multiples_of_256kb = int(in_bytes // (256 * 1024))
        return in_multiples_of_256kb * 256 * 1024

    def __call__(self, df: Pandas | Polars, *parts: str) -> tuple[()]:
        """Write a pandas or polars dataframe to Google Cloud Storage.

        Parameters
        ----------
        df: DataFrame
            The pandas or polars dataframe to upload.
        *parts: str
            Fragments that will be interpolated into the `prefix` given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `prefix`.

        Returns
        -------
        tuple
            An empty tuple.

        """

        key = self.prefix.format(*parts).strip(' /')

        client = self.gcs()
        bucket = client.get_bucket(self.bucket)
        blob = bucket.blob(key)

        if blob.exists():
            if self.skip:
                return ()
            if not self.overwrite:
                tmp = 'Object "{}" already exists in bucket "{}"!'
                msg = tmp.format(key, self.bucket)
                raise GcsError(msg)

        with blob.open('wb', self.chunk_bytes) as buffer:
            getattr(df, 'to_parquet', df.write_parquet)(buffer, **self.kwargs)

        return ()

    @staticmethod
    def __valid(chunk_size: Any) -> int:
        """Try to convert chunk_size to a meaningful integer."""
        try:
            as_int = int(chunk_size)
        except (TypeError, ValueError) as error:
            cls = type(chunk_size).__name__
            tmp = '"{}" must at least be convertible to integer, unlike {}!'
            msg = tmp.format('chunk_size', cls)
            raise TypeError(msg) from error
        if as_int < 1:
            tmp = '"{}" must be greater than (or equal to) one, unlike {}!'
            msg = tmp.format('chunk_size', as_int)
            raise ValueError(msg)
        return as_int

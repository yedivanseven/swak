from typing import Any
from io import BytesIO
from collections.abc import Callable
from functools import cached_property
from botocore.client import BaseClient
import pandas as pd
import polars as pl
from ...misc import ArgRepr, Bears, LiteralBears
from .s3 import S3


class S3Parquet2DataFrame[T](ArgRepr):
    """Download a single parquet file from S3 object storage.

    Type-annotate classes on instantiation with either a pandas or a polars
    dataframe so that static type checkers can infer the return type of
    the callable instances!

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The name of the bucket to upload to.
    prefix: str, optional
        The prefix of the parquet file to download. Since it (or part of it)
        can also be provided later, when the callable instance is called, it
        is optional here. Defaults to an empty string.
    bear: str, optional
        Type of dataframe to return. Can be one of "pandas" or "polars". Use
        the ``Bears`` enum to avoid typos. Defaults to "pandas".
    get_kws: dict, optional
        Keyword arguments (in addition to `Bucket` and `Key`) to pass to the
        `get_object <https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/client/get_object.html>`_ method of the client.
        Defaults to ``None``.
    **kwargs
        Additional keyword arguments are passed on to the top-level
        ``read_parquet`` function of either pandas or polars.

    See Also
    --------
    S3
    ~swak.misc.Bears

    """

    def __init__(
            self,
            s3: S3,
            bucket: str,
            prefix: str = '',
            bear: str | Bears | LiteralBears = Bears.PANDAS,
            get_kws: dict[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip().lstrip('/')
        self.bear = bear.strip().lower()
        self.get_kws = {} if get_kws is None else get_kws
        self.kwargs = kwargs
        super().__init__(
            self.s3,
            self.bucket,
            self.prefix,
            self.bear,
            get_kws=get_kws,
            **self.kwargs
        )

    @cached_property
    def client(self) -> BaseClient:
        """A cached instance of a fully configured S3 client."""
        return self.s3.client

    @property
    def read_parquet(self) -> Callable[[BytesIO, ...], T]:
        """Top-level ``read_parquet`` function of either pandas or polars."""
        return {
            Bears.PANDAS: pd.read_parquet,
            Bears.POLARS: pl.read_parquet
        }[self.bear]

    def __call__(self, path: str = '') -> T:
        """Download a single parquet file from S3 object storage.

        Parameters
        ----------
        path:
            The path to the parquet file to load. If given here, it will
            be appended to the `prefix` given at instantiation time.
            Defaults to an empty string.

        Returns
        -------
        DataFrame
            A pandas or polars dataframe, depending on `bear`.

        """
        stripped = path.strip(' /')
        prepended = '/' + stripped if stripped else stripped
        key = self.prefix + prepended
        response = self.client.get_object(
            Key=key,
            Bucket=self.bucket,
            **self.get_kws
        )
        with BytesIO(response.get('Body').read()) as buffer:
            df = self.read_parquet(buffer, **self.kwargs)
        return df

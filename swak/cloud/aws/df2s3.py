from typing import Any
from io import BytesIO
from botocore.client import BaseClient
from boto3.s3.transfer import TransferConfig
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
from ...misc import ArgRepr
from .s3 import S3


# ToDo: Add skip and overwrite. Act accordingly!
class DataFrame2S3Parquet(ArgRepr):
    """Upload a pandas or polars dataframe to an S3 bucket.

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The name of the bucket to upload to.
    prefix: str, optional
        The prefix of the parquet file to upload the dataframe to. May include
        any number of string placeholders (i.e., pairs of curly brackets) that
        will be interpolated when instances are called.
        Defaults to an empty string.
    extra_kws: dict, optional
        Passed on as ``ExtraArgs`` to the `upload_fileobj <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all options.
    upload_kws: dict, optional
        Passed on as ``Config`` to the `upload_fileobj <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all `options <options_>`__.
    **kwargs
        Additional keyword arguments are passed on to the ``to_parquet`` method
        of the dataframe.

    .. _meth: https://boto3.amazonaws.com/v1/documentation/api/latest/reference
        /services/s3/client/upload_fileobj.html
    .. _doc: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/customizations/s3.html#boto3.s3.transfer.TransferConfig
    .. _options: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/client/put_object.html

    See Also
    --------
    S3

    """

    def __init__(
            self,
            s3: S3,
            bucket: str,
            prefix: str = '',
            extra_kws: dict[str, Any] | None = None,
            upload_kws: dict[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip().lstrip('/')
        self.extra_kws = {} if extra_kws is None else extra_kws
        self.upload_kws = {} if upload_kws is None else upload_kws
        self.kwargs = kwargs
        super().__init__(
            self.s3,
            self.bucket,
            self.prefix,
            extra_kws=extra_kws,
            upload_kws=upload_kws,
            **self.kwargs
        )

    @property
    def client(self) -> BaseClient:
        """A cached instance of a fully configured S3 client."""
        return self.s3.client

    def __call__(self, df: Pandas | Polars, *parts: str) -> tuple[()]:
        """Write a pandas or polars dataframe to S3 object storage.

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
        key = self.prefix.format(*parts).strip()
        with BytesIO() as buffer:
            df.to_parquet(buffer, **self.kwargs)
            buffer.seek(0)
            self.client.upload_fileobj(
                Fileobj=buffer,
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=self.extra_kws,
                Config=TransferConfig(**self.upload_kws)
            )
        return ()

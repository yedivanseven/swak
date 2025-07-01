from typing import Any
from io import BytesIO
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from pandas import DataFrame as Pandas
from polars import DataFrame as Polars
from ...misc import ArgRepr
from .exceptions import S3Error
from .clients import S3


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
    overwrite: bool, optional
        Whether to silently overwrite the destination blob on S3. Defaults
        to ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the destination blob on S3 already
        exists. Defaults to ``False``.
    extra_kws: dict, optional
        Passed on as ``ExtraArgs`` to the `upload_fileobj <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all options.
    upload_kws: dict, optional
        Passed on as ``Config`` to the `upload_fileobj <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all `options <options_>`__.
    **kwargs
        Additional keyword arguments are passed on to the ``to_parquet``
        or ``write_parquet`` method of the dataframe.


    .. _meth: https://boto3.amazonaws.com/v1/documentation/api/latest/reference
        /services/s3/client/upload_fileobj.html
    .. _doc: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/customizations/s3.html#boto3.s3.transfer.TransferConfig
    .. _options: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/client/put_object.html

    Raises
    ------
    AttributeError
        If either `bucket` or `prefix` are not, in fact, strings.

    See Also
    --------
    S3

    """

    def __init__(
            self,
            s3: S3,
            bucket: str,
            prefix: str = '',
            overwrite: bool = False,
            skip: bool = False,
            extra_kws: dict[str, Any] | None = None,
            upload_kws: dict[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip(' /')
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.extra_kws = {} if extra_kws is None else extra_kws
        self.upload_kws = {} if upload_kws is None else upload_kws
        self.kwargs = kwargs
        super().__init__(
            self.s3,
            self.bucket,
            self.prefix,
            self.overwrite,
            self.skip,
            extra_kws=extra_kws,
            upload_kws=upload_kws,
            **self.kwargs
        )

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
        key = self.prefix.format(*parts).strip(' /')

        client = self.s3()

        try:
            _ = client.head_object(Bucket=self.bucket, Key=key)
            object_exists = True
        except ClientError:
            object_exists = False

        if object_exists:
            if self.skip:
                client.close()
                return ()
            if not self.overwrite:
                client.close()
                tmp = 'Object "{}" already exists in bucket "{}"!'
                msg = tmp.format(key, self.bucket)
                raise S3Error(msg)

        writer = 'to_parquet' if hasattr(df, 'to_parquet') else 'write_parquet'
        with BytesIO() as buffer:
            getattr(df, writer)(buffer, **self.kwargs)
            buffer.seek(0)
            try:
                client.upload_fileobj(
                    Fileobj=buffer,
                    Bucket=self.bucket,
                    Key=key,
                    ExtraArgs=self.extra_kws,
                    Config=TransferConfig(**self.upload_kws)
                )
            finally:
                client.close()

        return ()

from typing import Any
from botocore.exceptions import ClientError
from ...misc import ArgRepr
from .exceptions import S3Error
from .s3 import S3


class S3ListObjectKeys(ArgRepr):
    """List all object keys from S3 object storage with a given prefix.

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The name of the bucket to list objects from.
    prefix: str, optional
        The prefix to filter object keys. Defaults to an empty string,
        which will list all objects in the bucket.
    page_size: int, optional
        The number of objects to fetch per API call. Must be between 1 and
        1000. Defaults to 1000 for optimal performance.
    extra_kws: dict, optional
        Additional keyword arguments passed to the `list_objects_v2` paginator.
        See the `boto3 docs <doc_>`__ for all available options.


    .. _doc: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/paginator/ListObjectsV2.html

    Raises
    ------
    AttributeError
        If `bucket` or `prefix` are not, in fact, strings.
    ValueError
        If `page_size` is not between 1 and 1000.

    See Also
    --------
    S3

    """

    def __init__(
        self,
        s3: S3,
        bucket: str,
        prefix: str = '',
        page_size: int = 1000,
        extra_kws: dict[str, Any] | None = None,
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip().lstrip('/')
        if not (1 <= page_size <= 1000):
            msg = f'page_size must be between 1 and 1000, got {page_size}'
            raise ValueError(msg)
        self.page_size = int(page_size)
        self.extra_kws = {} if extra_kws is None else extra_kws
        super().__init__(
            self.s3, self.bucket, self.prefix, self.page_size, self.extra_kws
        )

    def __call__(self, additional_prefix: str = '') -> list[str]:
        """List all object keys from S3 object storage.

        Parameters
        ----------
        additional_prefix: str, optional
            Additional prefix to append to the configured prefix. Defaults
            to an empty string.

        Returns
        -------
        list[str]
            List of all object keys matching the prefix criteria.

        Raises
        ------
        S3Error
            If the bucket does not exist or other S3-related errors occur.

        """
        stripped = additional_prefix.strip(' /')
        full_prefix = self.prefix
        if stripped:
            full_prefix = (
                full_prefix + '/' + stripped if full_prefix else stripped
            )

        client = self.s3()
        try:
            paginator = client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=full_prefix,
                PaginationConfig={'PageSize': self.page_size},
                **self.extra_kws,
            )

            keys = []
            for page in page_iterator:
                if 'Contents' in page:
                    keys.extend(obj['Key'] for obj in page['Contents'])

            return keys

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                msg = f'Bucket "{self.bucket}" does not exist!'
                raise S3Error(msg) from e
            raise S3Error(f'S3 error: {e}') from e
        finally:
            client.close()

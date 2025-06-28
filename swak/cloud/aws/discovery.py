from typing import Any
from ...misc import ArgRepr
from .clients import S3


class S3ObjectDiscovery(ArgRepr):
    """List all files on S3 object storage with given a prefix and/or suffix.

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The name of the bucket to list files from.
    prefix: str, optional
        The prefix to filter file names by. Since it (or part of it)
        can also be provided later, when the callable instance is called, it
        is optional here. Defaults to an empty string.
    suffix: str, optional
        The suffix to filter file names by. Defaults to an empty string,
        which will allow all and any suffixes.
    subdir: bool, optional
        Whether to include files in "subdirectories", that is, files whose
        name contains more forward slashes than the specified `prefix`.
        Defaults to ``False``
    page_size: int, optional
        The number of file names to fetch per API call. Must be between 1 and
        1000. Defaults to 1000.
    **kwargs
        Additional keyword arguments passed to the `list_objects_v2` paginator.
        See the `boto3 docs <doc_>`__ for all available options.


    .. _doc: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/paginator/ListObjectsV2.html

    Raises
    ------
    AttributeError
        If `bucket`, `prefix`, or 'suffix' are not, in fact, strings.
    TypeError
        If `page_size` cannot be cast to an integer.
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
            suffix: str = '',
            subdir: bool = False,
            page_size: int = 1000,
            **kwargs: Any
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip().lstrip('/')
        self.suffix = suffix.strip().removeprefix('.')
        self.suffix = '.' + self.suffix if self.suffix else self.suffix
        self.subdir = bool(subdir)
        self.page_size = self.__valid(page_size)
        self.kwargs = kwargs
        super().__init__(
            self.s3,
            self.bucket,
            self.prefix,
            self.suffix,
            self.subdir,
            self.page_size,
            **kwargs
        )

    def __call__(self, path: str = '') -> list[str]:
        """List all files on S3 object storage that match the cached criteria.

        Parameters
        ----------
        path: str, optional
            The path to the files to list. If given here, it will be appended
            to the `prefix` given at instantiation time, separated by a
            forward slash. Defaults to an empty string.

        Returns
        -------
        list
           File names matching the prefix/suffix/subdir criteria.

        """
        stripped = path.strip().lstrip('/')
        if self.prefix and not self.prefix.endswith('/') and stripped:
            seperator = '/'
        else:
            seperator = ''
        prefix = self.prefix + seperator + stripped
        depth = prefix.count('/')

        client = self.s3()

        keys = []
        paginator = client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=prefix,
            PaginationConfig={'PageSize': self.page_size},
            **self.kwargs,
        )
        for page in pages:
            keys.extend(
                obj['Key']
                for obj in page.get('Contents', [])
                if obj['Key'].endswith(self.suffix)
                and (
                    obj['Key'].count('/') >= depth
                    if self.subdir else
                    obj['Key'].count('/') == depth
                )
            )

        client.close()

        return keys

    @staticmethod
    def __valid(page_size: int) -> int:
        """Try to convert page_size to a meaningful integer."""
        try:
            as_int = int(page_size)
        except (TypeError, ValueError) as error:
            cls = type(page_size).__name__
            tmp = '"{}" must at least be convertible to integer, unlike {}!'
            msg = tmp.format('page_size', cls)
            raise TypeError(msg) from error
        if not (1 <= as_int <= 1000):
            tmp = '"{}" must lie between 1 and 1000, unlike {}!'
            msg = tmp.format('page_size', as_int)
            raise ValueError(msg)
        return as_int

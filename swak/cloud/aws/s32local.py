import shutil
from typing import Any
from pathlib import Path
from tempfile import NamedTemporaryFile
from boto3.s3.transfer import TransferConfig
from ...misc import ArgRepr
from .exceptions import S3Error
from .clients import S3


# ToDo: Change to fsspec move.
class S3File2LocalFile(ArgRepr):
    """Download a single file from S3 object storage to local disk.

    Parameters
    ----------
    s3: S3
        An instance of a wrapped S3 client.
    bucket: str
        The name of the bucket to download the file from.
    prefix: str, optional
        The prefix of the file to download. Since it (or part of it) can also
        be provided later, when the callable instance is called, it is optional
        here. Defaults to an empty string.
    base_dir: str, optional
        Path to the directory on the local filesystem where the downloaded file
        should be saved. Defaults to the current working directory of the
        python interpreter.
    overwrite: bool, optional
        Whether to silently overwrite the local destination file. Defaults
        to ``False``, which will raise an exception if it already exists.
    skip: bool, optional
        Whether to silently do nothing if the local destination file already
        exists. Defaults to ``False``.
    extra_kws: dict, optional
        Passed on as ``ExtraArgs`` to the `download_file <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all options.
    download_kws: dict, optional
        Passed on as ``Config`` to the `download_file <meth_>`__ method of
        the client. See the `docs <doc_>`__ for all `options <options_>`__.


    .. _meth: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/client/download_fileobj.html
    .. _doc: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/customizations/s3.html#boto3.s3.transfer.TransferConfig
    .. _options: https://boto3.amazonaws.com/v1/documentation/api/latest/
        reference/services/s3/client/get_object.html

    Raises
    ------
    AttributeError
        If `bucket`, `prefix`, or `base_dir` are not, in fact, strings.

    See Also
    --------
    S3

    """

    def __init__(
            self,
            s3: S3,
            bucket: str,
            prefix: str = '',
            base_dir: str = '',
            overwrite: bool = False,
            skip: bool = False,
            extra_kws: dict[str, Any] | None = None,
            download_kws: dict[str, Any] | None = None,
    ) -> None:
        self.s3 = s3
        self.bucket = bucket.strip(' /')
        self.prefix = prefix.strip(' /')
        self.base_dir = str(Path(base_dir.strip()).resolve()).rstrip('/')
        self.overwrite = bool(overwrite)
        self.skip = bool(skip)
        self.extra_kws = {} if extra_kws is None else extra_kws
        self.download_kws = {} if download_kws is None else download_kws
        super().__init__(
            self.s3,
            self.bucket,
            self.prefix,
            self.base_dir,
            self.overwrite,
            self.skip,
            self.extra_kws,
            self.download_kws
        )

    def __call__(self, path: str = '') -> str:
        """Download a single file from S3 object storage.

        Parameters
        ----------
        path:
            The path to the file to load. If given here, it will be appended to
            both the `prefix` and the `base_dir` given at instantiation time.
            Defaults to an empty string.

        Returns
        -------
        str
            The fully resolved path of the downloaded file on local disk.

        """
        stripped = path.strip(' /')
        remote_separator = '/' if (self.prefix and stripped) else ''
        remote = self.prefix + remote_separator + stripped
        local_separator = '/' if (stripped or remote) else ''
        local = self.base_dir + local_separator + (stripped or remote)
        path = Path(local)

        if path.exists():
            if self.skip:
                return local
            if not self.overwrite:
                msg = f'File "{local}" already exists on local disk!'
                raise S3Error(msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        client = self.s3()

        with NamedTemporaryFile('wb') as file:
            try:
                client.download_fileobj(
                    Bucket=self.bucket,
                    Key=remote,
                    Fileobj=file,
                    ExtraArgs=self.extra_kws,
                    Config=TransferConfig(**self.download_kws)
                )
                shutil.move(file.name, path)
            finally:
                client.close()

        return local

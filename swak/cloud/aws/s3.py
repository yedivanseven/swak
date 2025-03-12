from typing import Any
import boto3
from botocore.client import BaseClient
from botocore.config import Config
from ...misc import ArgRepr


class S3(ArgRepr):
    """Wraps an S3 client for delayed instantiation and config encapsulation.

    For more convenient function chaining in functional compositions, instances
    are also callable and simply return a new client when called.

    Parameters
    ----------
    region_name: str, optional
        The name of the region associated with the client.
        Defaults to ``None``.
    api_version: str, optional
        The API version to use.  By default, botocore will use the latest API
        version when creating a client. You only need to specify this parameter
        if you want to use a previous API version of the client.
        Defaults to ``None``.
    use_ssl: bool, optional
        Whether to use SSL.  Defaults to ``True``.
    verify: bool or str, optional
        Whether to verify SSL certificates. Can be set to ``True``,
        a path/to/cert/bundle.pem, or ``False``. Defaults to ``True``.
    endpoint_url: str, optional
        The complete URL to use for the constructed client. Normally, botocore
        will automatically construct the appropriate URL to use when
        communicating with a service. You can specify a complete URL
        (including the "http/https" scheme) to override this behavior. If this
        value is provided, then `use_ssl` is ignored.
    aws_account_id: str, optional
        The account id to use when creating the client. Defaults to ``None``.
    aws_access_key_id: str, optional
        The access key to use when creating the client. Defaults to ``None``.
    aws_secret_access_key: str, optional
        The secret key to use when creating the client. Defaults to ``None``.
    aws_session_token: str, optional
        The session token to use for the client. Defaults to ``None``.
    **kwargs
        Additional parameters for the client. See the documentation of
        `context <https://boto3.amazonaws.com/v1/documentation/api/latest/
        guide/configuration.html#configuring-client-context-parameters>`_ and
        `config <https://botocore.amazonaws.com/v1/documentation/api/latest/
        reference/config.html>`_ for options. Defaults to ``None``.

    """

    def __init__(
            self,
            region_name: str | None = None,
            api_version: str | None = None,
            use_ssl: bool = True,
            verify: bool | str = True,
            endpoint_url: str | None = None,
            aws_account_id: str | None = None,
            aws_access_key_id: str | None = None,
            aws_secret_access_key: str | None = None,
            aws_session_token: str | None = None,
            **kwargs: Any
    ) -> None:
        self.region_name = self.__strip(region_name)
        self.api_version = self.__strip(api_version)
        self.use_ssl = use_ssl
        self.verify = verify.strip() if isinstance(verify, str) else verify
        self.endpoint_url = self.__strip(endpoint_url)
        self.__aws_account_id = self.__strip(aws_account_id)
        self.__aws_access_key_id = self.__strip(aws_access_key_id)
        self.__aws_secret_access_key = self.__strip(aws_secret_access_key)
        self.__aws_session_token = self.__strip(aws_session_token)
        self.kwargs = kwargs
        super().__init__(
            self.region_name,
            self.api_version,
            self.use_ssl,
            self.verify,
            self.endpoint_url,
            aws_account_id=self.aws_account_id,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            **self.kwargs
        )

    @staticmethod
    def __strip(attr: str | None) -> str:
        """Strip leading and trailing whitespaces from string arguments."""
        return attr if attr is None else attr.strip()

    @property
    def aws_account_id(self) -> str:
        """Obfuscated string representation of the AWS account ID."""
        return None if self.__aws_account_id is None else '****'

    @property
    def aws_access_key_id(self) -> str:
        """Obfuscated string representation of the AWS access key ID."""
        return None if self.__aws_access_key_id is None else '****'

    @property
    def aws_secret_access_key(self) -> str:
        """Obfuscated string representation of the secret AWS access key."""
        return None if self.__aws_secret_access_key is None else '****'

    @property
    def aws_session_token(self) -> str:
        """Obfuscated string representation of the AWS session token."""
        return None if self.__aws_session_token is None else '****'

    @property
    def client(self) -> BaseClient:
        """A fresh, new  S3 client."""
        return boto3.client(
            service_name='s3',
            region_name=self.region_name,
            api_version=self.api_version,
            use_ssl=self.use_ssl,
            verify=self.verify,
            endpoint_url=self.endpoint_url,
            aws_account_id=self.__aws_account_id,
            aws_access_key_id=self.__aws_access_key_id,
            aws_secret_access_key=self.__aws_secret_access_key,
            aws_session_token=self.__aws_session_token,
            config=Config(**self.kwargs)
        )

    def __call__(self, *_: Any, **__: Any) -> BaseClient:
        """Return a fresh, new S3 client, ignoring any (keyword) arguments."""
        return self.client

from typing import Any
from io import BytesIO
from functools import cached_property
from pandas import DataFrame
from swak.misc import ArgRepr
from botocore.client import BaseClient
from botocore.config import Config
import pandas as pd
import boto3


class S3Parquet2DataFrame(ArgRepr):

    def __init__(
            self,
            bucket: str,
            prefix: str = '',
            region_name: str | None = None,
            api_version: str | None = None,
            use_ssl: bool = True,
            verify: bool | str = True,
            endpoint_url: str | None = None,
            aws_account_id: str | None = None,
            aws_access_key_id: str | None = None,
            aws_secret_access_key: str | None = None,
            aws_session_token: str  | None = None,
            config: dict[str, Any] | None = None,
            get_kws: dict[str, Any] | None = None,
            **kwargs: Any
    ) -> None:
        self.bucket = self.__strip(bucket)
        self.prefix = self.__strip(prefix)
        self.region_name = self.__strip(region_name)
        self.api_version = self.__strip(api_version)
        self.use_ssl = use_ssl
        self.verify = verify.strip() if isinstance(verify, str) else verify
        self.endpoint_url = self.__strip(endpoint_url)
        self.__aws_account_id = self.__strip(aws_account_id)
        self.__aws_access_key_id = self.__strip(aws_access_key_id)
        self.__aws_secret_access_key = self.__strip(aws_secret_access_key)
        self.__aws_session_token = self.__strip(aws_session_token)
        self.config = {} if config is None else config
        self.get_kws = {} if get_kws is None else get_kws
        self.kwargs = kwargs
        super().__init__(
            self.bucket,
            self.prefix,
            self.region_name,
            self.api_version,
            self.use_ssl,
            self.verify,
            self.endpoint_url,
            aws_account_id=self.aws_account_id,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config=self.config,
            get_kws=self.get_kws,
            **self.kwargs
        )

    @staticmethod
    def __strip(attr: str | None) -> str:
        return attr if attr is None else attr.strip()

    @property
    def aws_account_id(self) -> str:
        return None if self.__aws_account_id is None else '****'

    @property
    def aws_access_key_id(self) -> str:
        return None if self.__aws_access_key_id is None else '****'

    @property
    def aws_secret_access_key(self) -> str:
        return None if self.__aws_secret_access_key is None else '****'

    @property
    def aws_session_token(self) -> str:
        return None if self.__aws_session_token is None else '****'

    @cached_property
    def client(self) -> BaseClient:
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
            config=Config(**self.config)
        )

    def __call__(self, path: str = '') -> DataFrame:
        stripped = path.strip().lstrip('/')
        prepended = '/' + stripped if stripped else stripped
        key = self.prefix + prepended
        response = self.client.get_object(
            Key=key,
            Bucket=self.bucket,
            **self.get_kws
        )
        with BytesIO(response.get('Body').read()) as buffer:
            df = pd.read_parquet(buffer, **self.kwargs)
        return df

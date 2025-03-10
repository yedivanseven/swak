from typing import Any
from io import BytesIO
from functools import cached_property
from pandas import DataFrame
from swak.misc import ArgRepr
from botocore.client import BaseClient
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
import boto3


class DataFrameS3Parquet(ArgRepr):

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
            extra_kws: dict[str, Any] | None = None,
            upload_kws: dict[str, Any] | None = None,
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
        self.extra_kws = {} if extra_kws is None else extra_kws
        self.upload_kws = {} if upload_kws is None else upload_kws
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
            extra_kws=self.extra_kws,
            upload_kws=self.upload_kws,
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

    def __call__(self, df: DataFrame, *parts: str) -> tuple[()]:
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

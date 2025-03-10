"""Tools to interact with elements of the Amazon Web Services (AWS).

Specifically, data scientists tend to interact heavily with S3 object storage.

"""

from .df2s3 import DataFrameS3Parquet
from .s32df import S3Parquet2DataFrame

__all__ = [
    'DataFrameS3Parquet',
    'S3Parquet2DataFrame'
]

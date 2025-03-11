"""Tools to interact with elements of the Amazon Web Services (AWS).

Specifically, data scientists tend to interact heavily with S3 object storage.

"""

from .s3 import S3
from .df2s3 import DataFrame2S3Parquet
from .s32df import S3Parquet2DataFrame

__all__ = [
    'S3',
    'DataFrame2S3Parquet',
    'S3Parquet2DataFrame'
]

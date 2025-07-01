"""Tools to interact with elements of the Amazon Web Services (AWS).

Specifically, data scientists tend to interact heavily with S3 object storage.

"""

from .clients import S3
from .bucket import S3Bucket
from .df2s3 import DataFrame2S3Parquet
from .s32df import S3Parquet2DataFrame
from .s32local import S3File2LocalFile
from .discovery import S3ObjectDiscovery

__all__ = [
    'S3',
    'S3Bucket',
    'DataFrame2S3Parquet',
    'S3Parquet2DataFrame',
    'S3File2LocalFile',
    'S3ObjectDiscovery'
]
# ToDo: Rewrite with Saver/Loader base class so that file/s3/gcs is handled

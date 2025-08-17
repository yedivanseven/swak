"""Tools to interact with elements of the Amazon Web Services (AWS).

Specifically, data scientists tend to interact heavily with S3 object storage.

"""

from .clients import S3
from .bucket import S3Bucket
from .s32local import S3File2LocalFile

__all__ = [
    'S3',
    'S3Bucket',
    'S3File2LocalFile'
]

"""Tools to interact with elements of the Amazon Web Services (AWS).

Specifically, data scientists tend to interact heavily with S3 object storage.

"""

from importlib.util import find_spec

if find_spec('boto3') is None:
    msg = 'Install {} with the [aws] extra to unlock this subpackage!'
    raise ImportError(msg.format(__package__.split('.')[0]))

from .clients import S3
from .bucket import S3Bucket

__all__ = [
    'S3',
    'S3Bucket'
]

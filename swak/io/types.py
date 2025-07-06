from enum import StrEnum
from typing import Any, Literal
from collections.abc import Mapping

type LiteralMode = Literal['wb', 'wt']
type LiteralStorage = Literal['file', 's3', 'gcs', 'memory']
type LiteralCompression = Literal['zip', 'bz2', 'gzip', 'lzma', 'xz']
type Toml = Mapping[str, Any]
type Yaml = Mapping[str, Any] | list[Any]


class Storage(StrEnum):
    """Supported file systems for read/write operations."""
    FILE = 'file'
    S3 = 's3'
    GCS = 'gcs'
    MEMORY = 'memory'


class Mode(StrEnum):
    """Modes for opening files."""
    WB = 'wb'
    WT = 'wt'


class Compression(StrEnum):
    """Compression algorithms for file storage."""
    ZIP = 'zip'
    BZ2 = 'bz2'
    GZIP = 'gzip'
    LZMA = 'lzma'
    XZ = 'xz'

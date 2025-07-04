"""Readers/Loaders and Writers/Savers for various file types and systems.

All (callable) classes are configured via a unified API. The flexibility of
switching the file system from local to, for example, a remote object storage
by changing a single argument ensures painless transitions between development,
staging, and production environments.

"""

from .writer import Writer, Storage, Mode
from .parquet import DataFrame2Parquet
from .toml import TomlWriter

__all__ = [
    'Writer',
    'Storage',
    'Mode',
    'DataFrame2Parquet',
    'TomlWriter'
]

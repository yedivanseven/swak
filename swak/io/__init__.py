"""Readers/Loaders and Writers/Savers for various file types and systems.

All (callable) classes are configured via a unified API. The flexibility of
switching the file system from local to, for example, a remote object storage
by changing a single argument ensures painless transitions between development,
staging, and production environments.

"""

from .types import Storage, Mode, Compression
from .writer import Writer
from .reader import Reader
from .parquet import DataFrame2Parquet
from .toml import TomlWriter, TomlReader
from .yaml import YamlWriter, YamlReader
from .json import JsonWriter

__all__ = [
    'Writer',
    'Reader',
    'Storage',
    'Mode',
    'Compression',
    'DataFrame2Parquet',
    'TomlWriter',
    'TomlReader',
    'YamlWriter',
    'YamlReader',
    'JsonWriter'
]

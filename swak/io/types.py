from typing import Any, Literal
from collections.abc import Mapping

type LiteralMode = Literal['wb', 'wt']
type LiteralStorage = Literal['file', 's3', 'gcs', 'memory']
type Toml = Mapping[str, Any]
type Yaml = Mapping[str, Any] | list[Any]

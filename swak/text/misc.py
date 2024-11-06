from typing import Any, Literal
from enum import StrEnum

type Toml = dict[str, Any]
type Yaml = dict[str, Any] | list[Any]
type LiteralNotFound = Literal['ignore', 'warn', 'raise']


class NotFound(StrEnum):
    """Enum to direct read/load behaviour in case of missing files.

    See Also
    --------
    TomlReader
    YamlReader

    """
    IGNORE = 'ignore'
    WARN = 'warn'
    RAISE = 'raise'

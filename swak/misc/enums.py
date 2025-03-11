from enum import StrEnum
from typing import Literal

type LiteralNotFound = Literal['ignore', 'warn', 'raise']


class NotFound(StrEnum):
    """Enum to direct read/load behaviour in case of missing files."""
    IGNORE = 'ignore'
    WARN = 'warn'
    RAISE = 'raise'

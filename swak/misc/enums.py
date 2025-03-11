from enum import StrEnum
from typing import Literal

type LiteralNotFound = Literal['ignore', 'warn', 'raise']
type LiteralBears = Literal['pandas', 'polars']


class NotFound(StrEnum):
    """Enum to direct read/load behaviour in case of missing files."""
    IGNORE = 'ignore'
    WARN = 'warn'
    RAISE = 'raise'


class Bears(StrEnum):
    """Enum to choose pandas versus polars."""
    PANDAS = 'pandas'
    POLARS = 'polars'

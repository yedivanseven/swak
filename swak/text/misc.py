from enum import StrEnum


class NotFound(StrEnum):
    """Enum to direct read/load behaviour in case of missing files.

    See Also
    --------
    TomlReader, YamlReader

    """
    IGNORE = 'ignore'
    WARN = 'warn'
    RAISE = 'raise'

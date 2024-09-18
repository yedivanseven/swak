import sys
import logging
from logging import Logger, Formatter, StreamHandler
from ..funcflow.loggers import DEFAULT_FMT


# ToDo: Write unit tests
def get_stdout_logger(
        name: str,
        level: int = logging.DEBUG,
        fmt: str = DEFAULT_FMT
) -> Logger:
    """Logger to stdout with at least one formatted StreamHandler.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.

    Notes
    -----
    To avoid creating and adding new Handler every time the same Logger is
    requested, one of the existing StreamHandlers will be modified if there
    are any. A new one will be created and added only if there aren't.
    By consequence, requesting the same Logger multiple times with a different
    `level` and/or a different `fmt` might change that Logger wherever it is
    used.

    """
    # Get logger with the given name
    logger = logging.getLogger(name)
    # The logging level can only be decreased but never increased
    logger.setLevel(max(min(level, logger.level), logging.DEBUG))
    # Prepare a formatter
    formatter = Formatter(fmt=fmt, style='{')
    formatter.default_msec_format = '%s.%03d'
    # Get existing StreamHandlers to stdout
    handlers = [
        handler for handler in logger.handlers
        if isinstance(handler, StreamHandler) and handler.stream is sys.stdout
    ]
    # Get the first matching handler if there is one or make a new one
    handler = handlers[0] if handlers else StreamHandler(sys.stdout)
    # The logging level can only be decreased but never increased
    handler.setLevel(max(min(level, handler.level), logging.DEBUG))
    # Set the format of the handler
    handler.setFormatter(formatter)
    # If the now configured handler was newly created, add it to the logger
    if not handlers:
        logger.addHandler(handler)
    # Now that we have at least one configured handler, return the logger
    return logger

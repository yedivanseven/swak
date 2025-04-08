import sys
import logging
from functools import cached_property
from logging import Logger, Formatter, StreamHandler, Handler
from ..repr import ArgRepr
from .formats import DEFAULT_FMT


class StdLogger(ArgRepr):
    """Wrapped logger to stdout or stderr with a formatted StreamHandler.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.
        Defaults to "{asctime:<23s} [{levelname:<8s}] {message} ({name})".
    stream: str, optional
        Must be one of "stdout" or "stderr". Defaults to "stdout".

    Raises
    ------
    TypeError
        If `stream` is not a string.
    ValueError
        If `stream` is neither "stdout" nor "stderr"

    Note
    ----
    The instantiation of the actual, underlying Logger is delayed until it is
    first needed to facilitate usage in multiprocessing scenarios.

    Warnings
    --------
    To avoid creating and adding a new Handler to the same stream every time
    the same Logger is requested, one of its existing StreamHandlers will be
    modified if there are any. A new one will be created and added only if
    there aren't. By consequence, requesting the same Logger multiple times
    with a different `level` and/or a different `fmt` might change that
    Logger globally.

    """
    def __init__(
            self,
            name: str,
            level: int = logging.DEBUG,
            fmt: str = DEFAULT_FMT,
            stream: str = 'stdout'
    ) -> None:
        self.name = name.strip()
        self.level = min(max(level, logging.DEBUG), logging.CRITICAL)
        self.fmt = fmt
        self.stream = self.__valid(stream)
        super().__init__(self.name, self.level, fmt, stream)

    @staticmethod
    def __valid(stream: str) -> str:
        """Ensure that the provided stream is one of the permitted options."""
        if not isinstance(stream, str):
            cls = type(stream).__name__
            msg = f'stream must be a string, not {cls}!'
            raise TypeError(msg)
        stream = stream.strip().lower()
        if stream not in ('stdout', 'stderr'):
            msg = f'stream must be "stdout" or "stderr", not "{stream}"!'
            raise ValueError(msg)
        return stream

    def log(self, level: int, message: str) -> tuple[()]:
        """Log a message at the given level.

        Parameters
        ----------
        level: int
            The level to log at.
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        self.logger.log(level, message)
        return ()

    def debug(self, message: str) -> tuple[()]:
        """Log a DEBUG message.

        Parameters
        ----------
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.DEBUG, message)

    def info(self, message: str) -> tuple[()]:
        """Log an INFO message.

        Parameters
        ----------
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.INFO, message)

    def warning(self, message: str) -> tuple[()]:
        """Log a WARNING message.

        Parameters
        ----------
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.WARNING, message)

    def error(self, message: str) -> tuple[()]:
        """Log an ERROR message.

        Parameters
        ----------
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.ERROR, message)

    def critical(self, message: str) -> tuple[()]:
        """Log a CRITICAL message.

        Parameters
        ----------
        message: str
            The message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.CRITICAL, message)

    @cached_property
    def logger(self) -> Logger:
        """The requested Logger with one StreamHandler configured to specs."""
        # Get Logger with the given name.
        logger = logging.getLogger(self.name)
        # Adjust its log level so that messages from the Handler get through.
        logger.setLevel(self.level)
        # Get all StreamHandlers to the requested stream.
        hdls = self.__filtered(logger.handlers)
        # If there are any, get the first one. If not, make a new one ...
        hdl = hdls[0] if hdls else StreamHandler(getattr(sys, self.stream))
        # ... and configure it to specs.
        configured = self.__configured(hdl)
        # If it was a new one, add it to the logger ...
        if not hdls:
            logger.addHandler(configured)
        # ... and return the logger.
        return logger

    def __filtered(self, handlers: list[Handler]) -> tuple[StreamHandler, ...]:
        """Filter the Logger's Handlers according to the filter criterion."""
        return tuple(filter(self.__handles_the_same_stream, handlers))

    def __handles_the_same_stream(self, handler: Handler) -> bool:
        """Is the Handler a StreamHandler to the requested stream?"""
        if isinstance(handler, StreamHandler):
            return handler.stream is getattr(sys, self.stream)
        return False

    def __configured(self, handler: StreamHandler) -> StreamHandler:
        """Configure the selected StreamHandler."""
        # Set the level of the handler
        handler.setLevel(self.level)
        # Set the configured formatter
        handler.setFormatter(self.__formatter)
        return handler

    @property
    def __formatter(self) -> Formatter:
        """LogRecord Formatter for the specified format."""
        formatter = Formatter(fmt=self.fmt, style='{')
        formatter.default_msec_format = '%s.%03d'
        return formatter

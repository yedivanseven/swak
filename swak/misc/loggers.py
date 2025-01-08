import sys
import logging
from pathlib import Path
from functools import cached_property
from logging import Logger, Formatter, StreamHandler, FileHandler, Handler
from .repr import ArgRepr

DEFAULT_FMT = '{asctime:<23s} [{levelname:<8s}] {message} ({name})'
PID_FMT = '{asctime:<23s} [{levelname:<8s}] {message} ({name} | PID-{process})'


class StdOutLogger(ArgRepr):
    """Wrapped logger to stdout with at least one formatted StreamHandler.

    The instantiation of the actual, underlying Logger is delayed until it is
    first needed to facilitate usage in multiprocessing scenarios.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.

    Warnings
    --------
    To avoid creating and adding a new Handler every time the same Logger is
    requested, one of its existing StreamHandlers will be modified if there
    are any. A new one will be created and added only if there aren't.
    By consequence, requesting the same Logger multiple times with a different
    `level` and/or a different `fmt` might change that Logger globally.

    """
    def __init__(
            self,
            name: str,
            level: int = logging.DEBUG,
            fmt: str = DEFAULT_FMT
    ) -> None:
        self.name = name.strip()
        self.level = level
        self.fmt = fmt
        super().__init__(self.name, level, fmt)

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
        """The specified Logger with one StreamHandler configured to specs."""
        # Get logger with the given name
        logger = logging.getLogger(self.name)
        # Adjust the logger level so that messages from the handler get through
        logger.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
        # Get StreamHandlers to stdout
        handlers = self.__filtered(logger.handlers)
        # Get the first matching handler if there is one or make a new one
        handler = handlers[0] if handlers else StreamHandler(sys.stdout)
        # Configure that handler according to specs
        configured = self.__configure(handler)
        # If the now configured handler was newly created, add it to the logger
        if not handlers:
            logger.addHandler(configured)
        return logger

    def __filtered(self, handlers: list[Handler]) -> tuple[StreamHandler, ...]:
        """Filter the Logger's Handlers according to the filter criterion."""
        return tuple(filter(self.__is_stdout_streamhandler, handlers))

    @staticmethod
    def __is_stdout_streamhandler(handler: Handler) -> bool:
        """Filter criterion for the Logger's Handlers."""
        if isinstance(handler, StreamHandler):
            return handler.stream is sys.stdout
        return False

    def __configure(self, handler: StreamHandler) -> StreamHandler:
        """Configure the selected StreamHandler to stdout."""
        # Set the level of the handler
        handler.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
        # Set the configured formatter
        handler.setFormatter(self.__formatter)
        return handler

    @property
    def __formatter(self) -> Formatter:
        """LogRecord Formatter for the specified format."""
        formatter = Formatter(fmt=self.fmt, style='{')
        formatter.default_msec_format = '%s.%03d'
        return formatter


class FileLogger(ArgRepr):
    """Wrapped logger to file with at least one formatted FileHandler.

    The instantiation of the actual, underlying Logger is delayed until it is
    first needed to facilitate usage in multiprocessing scenarios.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    file: str
        Name of the file to log to, including file extension.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.
    mode: str, optional
        Mode to open the file. Defaults to 'a'.
    encoding: str, optional
        Encoding to use when opening the file. Defaults to 'utf-8'.
    delay: bool, optional
        Whether to delay opening the file until it is first written to.
        Defaults to ``True``.

    Warnings
    --------
    To avoid creating and adding a new Handler to the same file every time
    the same Logger is requested, one of its existing FileHandlers (to the
    specified file) will be modified if there are any. A new one will be
    created and added only if there aren't. By consequence, requesting the
    same Logger multiple times with a different `level` and/or a different
    `fmt` might change that Logger globally.

    """
    def __init__(
            self,
            name: str,
            file: str,
            level: int = logging.DEBUG,
            fmt: str = DEFAULT_FMT,
            mode: str = 'a',
            encoding: str = 'utf-8',
            delay: bool = True,
    ) -> None:
        self.name = name.strip()
        self.file = str(Path(file.strip()).resolve())
        self.level = level
        self.fmt = fmt
        self.mode = mode.strip()
        self.encoding = encoding.strip()
        self.delay = delay
        super().__init__(
            self.name,
            self.file,
            level,
            fmt,
            self.mode,
            self.encoding,
            self.delay
        )

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
        """The specified Logger with one FileHandler configured to specs."""
        # Get logger with the given name
        logger = logging.getLogger(self.name)
        # Adjust the logger level so that messages from the handler get through
        logger.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
        # Get all FileHandlers to the specified file
        handlers = self.__filtered(logger.handlers)
        # Get the first matching handler if there is one or make a new one
        handler = handlers[0] if handlers else FileHandler(
            self.file,
            self.mode,
            self.encoding,
            self.delay
        )
        # Configure that handler according to specs
        configured = self.__configure(handler)
        # If the now configured handler was newly created, add it to the logger
        if not handlers:
            logger.addHandler(configured)
        return logger

    def __filtered(self, handlers: list[Handler]) -> tuple[FileHandler, ...]:
        """Filter the Logger's Handlers according to the filter criterion."""
        return tuple(filter(self.__is_matching_filehandler, handlers))

    def __is_matching_filehandler(self, handler: Handler) -> bool:
        """Filter criterion for the Logger's Handlers."""
        if isinstance(handler, FileHandler):
            return handler.baseFilename == self.file
        return False

    def __configure(self, handler: FileHandler) -> FileHandler:
        """Configure the selected FileHandler."""
        # Set the level of the handler
        handler.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
        # Set the configured formatter
        handler.setFormatter(self.__formatter)
        return handler

    @property
    def __formatter(self) -> Formatter:
        """LogRecord Formatter for the specified format."""
        formatter = Formatter(fmt=self.fmt, style='{')
        formatter.default_msec_format = '%s.%03d'
        return formatter

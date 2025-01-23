import logging
from pathlib import Path
from functools import cached_property
from logging import Logger, Formatter, FileHandler, Handler
from ..repr import ArgRepr
from .formats import DEFAULT_FMT


class FileLogger(ArgRepr):
    """Wrapped logger to file with at least one formatted FileHandler.

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

    Raises
    ------
    FileExistsError
        If a different Logger already has a FileHandler to the very same file.
        Because the instantiation of the actual Logger is delayed until it is
        first needed (to facilitate usage in multiprocessing scenarios),
        raising the exception is also delayed.

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
            self.mode,
            level,
            fmt,
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
        # No two FileHandlers should handle the same file
        if self.handler_exists:
            msg = f'A different logger already handles file "{self.file}"!'
            raise FileExistsError(msg)
        # Adjust the logger level so that messages from the handler get through
        logger.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
        # Create a new file handler according to specs, ...
        handler = FileHandler(self.file, self.mode, self.encoding, self.delay)
        # ... configure it, add it to the logger ...
        logger.addHandler(self.__configured(handler))
        # ... and return the logger.
        return logger

    @property
    def handler_exists(self) -> bool:
        """Does a Logger with a Handler of the specified file already exist?"""
        root = logging.getLogger()
        candidates = [root] + list(root.manager.loggerDict.values())
        loggers = filter(lambda obj: isinstance(obj, Logger), candidates)
        return any(
            any(self.__handles_file(handler) for handler in logger.handlers)
            for logger in loggers
        )

    def __handles_file(self, handler: Handler) -> bool:
        """Is the Handler a FileHandler to the requested file?"""
        if isinstance(handler, FileHandler):
            return handler.baseFilename == self.file
        return False

    def __configured(self, handler: FileHandler) -> FileHandler:
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

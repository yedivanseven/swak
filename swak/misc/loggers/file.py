import logging
from pathlib import Path
from functools import cached_property
from logging import Logger, Formatter, FileHandler, Handler
from ..repr import ArgRepr
from .formats import SHORT_FMT


class FileLogger(ArgRepr):
    """Wrapped logger to file with at least one formatted FileHandler.

    Parameters
    ----------
    file: str
        Full path to the file to log to, including file extension. If the
        parent directory does not exist, an attempt will be made to create it.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.
        Defaults to "{asctime:<23s} [{levelname:<8s}] {message}".
    mode: str, optional
        Mode to open the file. Defaults to "a".
    encoding: str, optional
        Encoding to use when opening the file. Defaults to "utf-8".
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

    Note
    ----
    The `mode` and the `encoding` can not be changed on an existing log file.

    """
    def __init__(
            self,
            file: str,
            level: int = logging.DEBUG,
            fmt: str = SHORT_FMT,
            mode: str = 'a',
            encoding: str = 'utf-8',
            delay: bool = True,
    ) -> None:
        self.file = str(Path(file.strip()).resolve())
        self.level = min(max(level, logging.DEBUG), logging.CRITICAL)
        self.fmt = fmt
        self.mode = mode.strip()
        self.encoding = encoding.strip()
        self.delay = delay
        super().__init__(
            self.file,
            self.mode,
            self.level,
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
        """The requested Logger with one FileHandler configured to specs."""
        # Check if some other Logger already has a Handler onto the same file.
        if self.handler_exists:
            msg = f'A different logger already handles file "{self.file}"!'
            raise FileExistsError(msg)
        # Get Logger with the given name.
        logger = logging.getLogger(self.name)
        # Adjust its log level so that messages from the Handler get through.
        logger.setLevel(self.level)
        # Get all Handlers to the requested file. There should be at most 1!
        handlers = tuple(filter(self.__handles_file, logger.handlers))
        # If there are any, ...
        if handlers:
            # ... get the first one, ...
            hdl = handlers[0]
            # ... set its delay attribute, and, ...
            hdl.delay = self.delay
            # ... if the file does not yet exist, ...
            if not Path(self.name).exists():
                # ... also mode and encoding.
                hdl.mode = self.mode
                hdl.encoding = self.encoding
        # If there aren't any handlers yet, make a new one and create directory
        else:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            hdl = FileHandler(self.file, self.mode, self.encoding, self.delay)
        # Configure the FileHandler to specs.
        configured = self.__configured(hdl)
        # If it was a new one, add it to the logger ...
        if not handlers:
            logger.addHandler(configured)
        # ... and return the logger.
        return logger

    @property
    def name(self) -> str:
        """The name given to the Logger based on the provided file name."""
        return self.file.replace('.', '_')

    @property
    def handler_exists(self) -> bool:
        """Does a Logger with a Handler of the specified file already exist?"""
        root = logging.getLogger()
        candidates = [root] + list(root.manager.loggerDict.values())
        loggers = filter(lambda obj: isinstance(obj, Logger), candidates)
        return any(
            any(self.__handles_file(handler) for handler in logger.handlers)
            for logger in loggers
            if logger.name != self.name
        )

    def __handles_file(self, handler: Handler) -> bool:
        """Is the Handler a FileHandler to the requested file?"""
        if isinstance(handler, FileHandler):
            return handler.baseFilename == self.file
        return False

    def __configured(self, handler: FileHandler) -> FileHandler:
        """Configure the selected FileHandler."""
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

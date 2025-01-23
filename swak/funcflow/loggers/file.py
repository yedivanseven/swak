import logging
from typing import ParamSpec
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from logging import Logger, Formatter, FileHandler, Handler
from ...misc import ArgRepr
from .formats import DEFAULT_FMT

P = ParamSpec('P')
type Message = str | Callable[P, str]


class PassThroughFileLogger(ArgRepr):
    """Pass-through Logger to file with at least one formatted FileHandler.

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

    @staticmethod
    def __valid(stream: str) -> str:
        """Ensure that the provided stream is one the permitted options."""
        if not isinstance(stream, str):
            cls = type(stream).__name__
            msg = f'stream must be a string, not {cls}!'
            raise TypeError(msg)
        stream = stream.strip().lower()
        if stream not in ('stdout', 'stderr'):
            msg = f'stream must be "stdout" or "stderr", not "{stream}"!'
            raise ValueError(msg)
        return stream

    class Log:
        """Return type of the logging methods.

        Note
        ----
        This class is not intended to ever be instantiated directly.

        See Also
        --------
        PassThroughFileLogger.debug
        PassThroughFileLogger.info
        PassThroughFileLogger.warning
        PassThroughFileLogger.error
        PassThroughFileLogger.critical

        """

        def __init__(
                self,
                parent: 'PassThroughFileLogger',
                level: int,
                msg: Message
        ) -> None:
            self.parent = parent
            self.level = level
            self.msg = msg

        def __call__(self, *args: P.args) -> P.args:
            """Log the cached message and pass through the input argument(s).

            Parameters
            ----------
            *args
                If `msg` is a string, these arguments are inconsequential,
                and it will simply be logged. If, however, the `msg` is
                callable, it will be called with these arguments and the
                return value will be logged instead.

            Returns
            -------
            object or tuple
                Single argument if called with only one, otherwise all of them.

            """
            msg = self.msg(*args) if callable(self.msg) else self.msg
            self.parent.logger.log(self.level, msg)
            return args[0] if len(args) == 1 else args

    def debug(self, message: Message) -> Log:
        """Log a DEBUG message, optionally depending on call arguments.

        Parameters
        ----------
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.DEBUG, message)

    def info(self, message: Message):
        """Log an INFO message, optionally depending on call arguments.

        Parameters
        ----------
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.INFO, message)

    def warning(self, message: Message):
        """Log a WARNING message, optionally depending on call arguments.

        Parameters
        ----------
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.WARNING, message)

    def error(self, message: Message):
        """Log an ERROR message, optionally depending on call arguments.

        Parameters
        ----------
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.ERROR, message)

    def critical(self, message: Message):
        """Log a CRITICAL message, optionally depending on call arguments.

        Parameters
        ----------
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.CRITICAL, message)

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

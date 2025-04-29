import sys
import logging
from typing import Literal
from collections.abc import Callable
from functools import cached_property
from logging import Logger, Formatter, StreamHandler, Handler
from ...misc import ArgRepr
from .formats import DEFAULT_FMT


class PassThroughStdLogger[**P](ArgRepr):
    """Pass-through Logger to stdout or stderr with a formatted StreamHandler.

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
            stream: Literal['stdout', 'stderr'] = 'stdout'
    ) -> None:
        self.name = name.strip()
        self.level = min(max(level, logging.DEBUG), logging.CRITICAL)
        self.fmt = fmt
        self.stream = self.__valid(stream)
        super().__init__(self.name, self.level, fmt, stream)

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
        PassThroughStdLogger.debug
        PassThroughStdLogger.info
        PassThroughStdLogger.warning
        PassThroughStdLogger.error
        PassThroughStdLogger.critical

        """

        def __init__(
                self,
                parent: 'PassThroughStdLogger',
                level: int,
                msg: str | Callable[P, str]
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

    def log(self, level: int, message: str | Callable[P, str]) -> Log:
        """Log a message a a level, optionally depending on call arguments.

        Parameters
        ----------
        level: int
            The level to log at.
        message: str or callable
            Message to log when the returned object is called. If it is
            callable, then it will be called with whatever argument(s) the
            returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, level, message)

    def debug(self, message: str | Callable[P, str]) -> Log:
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

    def info(self, message: str | Callable[P, str]) -> Log:
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

    def warning(self, message: str | Callable[P, str]) -> Log:
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

    def error(self, message: str | Callable[P, str]) -> Log:
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

    def critical(self, message: str | Callable[P, str]) -> Log:
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

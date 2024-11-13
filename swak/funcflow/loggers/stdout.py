import sys
import logging
from typing import ParamSpec
from collections.abc import Callable
from functools import cached_property
from logging import Logger, Formatter, StreamHandler, Handler
from ...misc import ArgRepr

P = ParamSpec('P')
type Message = str | Callable[P, str]

DEFAULT_FMT = '{asctime:<23s} [{levelname:<8s}] {message} ({name})'
PID_FMT = '{asctime:<23s} [{levelname:<8s}] {message} ({name} | PID-{process})'


class PassThroughStdOut(ArgRepr):
    """Pass-through Logger to stdout with at least one formatted StreamHandler.

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
    `level` and/or a different `fmt` will change that Logger globally.

    """
    def __init__(
            self,
            name: str,
            level: int = logging.DEBUG,
            fmt: str = DEFAULT_FMT
    ) -> None:
        super().__init__(name, level, fmt)
        self.name = name
        self.level = level
        self.fmt = fmt

    class Log:
        """Return type of the logging methods.

        Note
        ----
        This class is not intended to ever be instantiated directly.

        See Also
        --------
        PassThroughStdOut.debug
        PassThroughStdOut.info
        PassThroughStdOut.warning
        PassThroughStdOut.error
        PassThroughStdOut.critical

        """

        def __init__(
                self,
                parent: 'PassThroughStdOut',
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

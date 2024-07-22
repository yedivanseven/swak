import sys
import logging
from typing import ParamSpec, Callable
from functools import cached_property
from logging import Logger, Formatter, StreamHandler, Handler
from ..magic import ArgRepr

P = ParamSpec('P')
type Message = str | Callable[P, str]

DEFAULT_FMT = '{asctime:<23s} [{levelname:<8s}] {message} ({name})'


class StdOut(ArgRepr):
    """Pass-through logger to stdout with at least one formatted StreamHandler.

    Parameters
    ----------
    name: str
        Name of the logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: str, optional
        Format string for the log messages in ``str.format()`` format.

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

        Notes
        -----
        This class is not intended to ever be instantiated directly.

        See Also
        --------
        StdOut.debug
        StdOut.info
        StdOut.warning
        StdOut.error
        StdOut.critical

        """

        def __init__(self, logger: Logger, level: int, msg: Message) -> None:
            self.logger = logger
            self.level = level
            self.msg = msg

        def __call__(self, *args: P.args) -> P.args:
            """Log the cached message and pass through the input argument(s).

            Parameters
            ----------
            *args
                If `msg` is a string, these arguments are inconsequential,
                and it will simply be logged. If, however, the `msg` is
                callable, it will be called with these arguments and its
                return value will be logged instead.

            Returns
            -------
            object or tuple
                Single argument if called with only one, otherwise all of them.

            """
            msg = self.msg(*args) if callable(self.msg) else self.msg
            self.logger.log(self.level, msg)
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
        return self.Log(self.logger, logging.DEBUG, message)

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
        return self.Log(self.logger, logging.INFO, message)

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
        return self.Log(self.logger, logging.WARNING, message)

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
        return self.Log(self.logger, logging.ERROR, message)

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
        return self.Log(self.logger, logging.CRITICAL, message)

    @cached_property
    def logger(self) -> Logger:
        """Create or get the specified logger and configure it to specs."""
        # Get logger with the given name
        logger = logging.getLogger(self.name)
        # The logging level can only be decreased but never increased
        logger.setLevel(max(min(self.level, logger.level), logging.DEBUG))
        # Get StreamHandlers to stdout
        handlers = self.__filtered(logger.handlers)
        # If there is at least one, configure the first according to specs
        if handlers:
            self.__configure(handlers[0])
        # If not, configure a new one and add it to the logger
        else:
            logger.addHandler(self.__configure(StreamHandler(sys.stdout)))
        return logger

    def __filtered(self, handlers: list[Handler]) -> tuple[StreamHandler]:
        """Filter log handlers according to the filter criterion."""
        return tuple(filter(self.__is_stdout_streamhandler, handlers))

    @staticmethod
    def __is_stdout_streamhandler(handler: Handler) -> bool:
        """Filter criterion for log handlers."""
        if isinstance(handler, StreamHandler):
            return handler.stream is sys.stdout
        return False

    def __configure(self, handler: StreamHandler) -> StreamHandler:
        """Configure the selected stdout log handler."""
        # The logging level can only be decreased but never increased
        handler.setLevel(max(min(self.level, handler.level), logging.DEBUG))
        handler.setFormatter(self.__formatter)
        return handler

    @property
    def __formatter(self) -> Formatter:
        """Log-record formatter for the specified format."""
        formatter = Formatter(fmt=self.fmt, style='{')
        formatter.default_msec_format = '%s.%03d'
        return formatter

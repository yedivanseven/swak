import sys
import logging
from typing import Literal, Any
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from logging import Logger, StreamHandler, Handler
from ...misc import ArgRepr, JsonStreamHandler
from .formats import JSON_FMT

type JSON = Mapping[str, Any]


class PassThroughJsonLogger[**P](ArgRepr):
    """Pass-through Logger to stdout or stderr with a JSON-formatted messages.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    stream: str, optional
        Must be one of "stdout" or "stderr". Defaults to "stdout".
    field: str or iterable, optional
        A single `LogRecord attribute <https://docs.python.org/3/library/
        logging.html#logrecord-attributes>`_ or an iterable thereof. All values
        that are not marked by "You shouldn’t need to format this yourself."
        are allowed. Logged messages will be merged with a dictionary of these
        keys before being written to `stream` unless you explicitly refer to
        the log-record attribute "message", in which case that field will
        contain the dictionary you want to log.
        Defaults to ``("levelname", "name")``.
    *fields: str
        Additional LogRecord attributes to merge with the log message.
    **extras
        Additional, static key-value pairs to merge into every log message.

    Note
    ----
    The instantiation of the actual, underlying Logger is delayed until it is
    first needed to facilitate usage in multiprocessing scenarios.

    Warnings
    --------
    If the requested Logger already has a Handler to the requested stream, it
    will be deleted and replaced with a new one. By consequence, requesting
    the same Logger multiple times with different options might change that
    Logger globally.

    """
    def __init__(
        self,
        name: str,
        level: int = logging.DEBUG,
        stream: Literal['stdout', 'stderr'] = 'stdout',
        field: str | Iterable[str] = JSON_FMT,
        *fields: str,
        **extras: Any,
    ) -> None:
        self.name = name.strip()
        self.level = min(max(level, logging.DEBUG), logging.CRITICAL)
        self.stream = stream
        self.field = field
        self.fields = fields
        self.extras = extras
        super().__init__(
            self.name,
            self.level,
            stream,
            field,
            *fields,
            **extras
        )

    class Log:
        """Return type of the logging methods.

        Note
        ----
        This class is not intended to ever be instantiated directly.

        See Also
        --------
        PassThroughJsonLogger.debug
        PassThroughJsonLogger.info
        PassThroughJsonLogger.warning
        PassThroughJsonLogger.error
        PassThroughJsonLogger.critical

        """

        def __init__(
                self,
                parent: 'PassThroughJsonLogger',
                level: int,
                msg: JSON | Callable[[JSON], str]
        ) -> None:
            self.parent = parent
            self.level = level
            self.msg = msg

        def __call__(self, *args: P.args) -> P.args:
            """Log the cached message and pass through the input argument(s).

            Parameters
            ----------
            *args
                If `msg` is dictionary-like, the arguments are inconsequential,
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

    def log(self, level: int, message: JSON | Callable[[P], JSON]) -> Log:
        """Log a (dictionary) message, optionally depending on call args.

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

    def debug(self, message: JSON | Callable[[P], JSON]) -> Log:
        """Log a DEBUG (dictionary) message, optionally depending on call args.

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

    def info(self, message: JSON | Callable[[P], JSON]) -> Log:
        """Log an INFO (dictionary) message, optionally depending on call args.

        Parameters
        ----------
        message: Mapping or callable
            (Dictionary) message to log when the returned object is called.
            If it is callable, then it will be called with whatever argument(s)
            the returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.INFO, message)

    def warning(self, message: JSON | Callable[[P], JSON]) -> Log:
        """Log a WARNING (dict) message, optionally depending on call args.

        Parameters
        ----------
        message: Mapping or callable
            (Dictionary) message to log when the returned object is called.
            If it is callable, then it will be called with whatever argument(s)
            the returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.WARNING, message)

    def error(self, message: JSON | Callable[[P], JSON]) -> Log:
        """Log an ERROR (dict-like) message, optionally depending on call args.

        Parameters
        ----------
        message: Mapping or callable
            (Dictionary) message to log when the returned object is called.
            If it is callable, then it will be called with whatever argument(s)
            the returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.ERROR, message)

    def critical(self, message: JSON | Callable[[P], JSON]) -> Log:
        """Log a CRITICAL (dict) message, optionally depending on call args.

        Parameters
        ----------
        message: Mapping or callable
            (Dictionary) message to log when the returned object is called.
            If it is callable, then it will be called with whatever argument(s)
            the returned object is called with and the result will be logged.

        Returns
        -------
        Log
            A callable object that logs the `message` when called.

        """
        return self.Log(self, logging.CRITICAL, message)

    @cached_property
    def logger(self) -> Logger:
        """The requested Logger with exactly one JsonStreamHandler."""
        # Get the Logger with the given name
        logger = logging.getLogger(self.name)
        # Adjust its log level so that messages from the Handler get through
        logger.setLevel(self.level)
        # Remove all existing Handlers to the requested stream
        for handler in self.__filtered(logger.handlers):
            logger.removeHandler(handler)
        # Create a new JsonStreamHandler
        handler = JsonStreamHandler(
            self.stream,
            self.field,
            *self.fields,
            **self.extras
        )
        handler.setLevel(self.level)
        # ... add the new one instead, ...
        logger.addHandler(handler)
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

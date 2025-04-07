import sys
import json
import logging
from typing import Any, Literal
from collections.abc import Callable, Iterable
from functools import cached_property
from datetime import datetime
from logging import Logger, Formatter, StreamHandler, Handler, LogRecord
from ..repr import ArgRepr
from .formats import JSON_FMT


class JsonLogger(ArgRepr):
    """Wrapped logger to stdout or stderr with a JSON-formatted StreamHandler.

    Parameters
    ----------
    name: str
        Name of the Logger. Typically set to ``__name__``.
    level: int, optional
        Minimum logging level. Defaults to 10 (= DEBUG).
    fmt: dict, optional
        Mapping with JSON field names as keys and the `log-record attributes
        <https://docs.python.org/3/library/logging.html#logrecord-attributes>`_
        they should contain as values. Logged messages will be merged
        with this mapping before being written to `stream` unless you
        explicitly refer to the log-record-attribute "message", in which case
        this field will contain the dictionary you want to log.
        Defaults to ``{"level": "levelname", "logger": "name"}``.
    stream: str, optional
        Must be one of "stdout" or "stderr". Defaults to "stdout".
    default: callable, optional
        Called on objects that cannot otherwise be JSON serialized.
        Defaults to python's builtin ``str()``.
    **kwargs
        Additional keyword arguments are passed on to the ``json.dumps``
        method. See its `documentation <https://docs.python.org/3/library/
        json.html#json.dump>`_ for details.

    Raises
    ------
    TypeError
        If `stream` is not a string or if `default` is not callable.
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
            fmt: dict[str, str] = JSON_FMT,
            stream: Literal['stdout', 'stderr'] = 'stdout',
            default: Callable = str,
            **kwargs: Any
    ) -> None:
        self.name = name.strip()
        self.level = level
        self.fmt = fmt
        self.stream = self.__valid(stream)
        self.default = self.__callable(default)
        self.kwargs = kwargs
        super().__init__(
            self.name,
            level,
            fmt,
            stream,
            self.default,
            **kwargs
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

    @staticmethod
    def __callable(default: Callable) -> Callable:
        if not callable(default):
            cls = type(default).__name__
            msg = f'default of type {cls} is not callable!'
            raise TypeError(msg)
        return default

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
        logger.setLevel(min(max(self.level, logging.DEBUG), logging.CRITICAL))
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


# ToDo: Write docstrings and unit tests
# ToDo: Explore how to combine with logger
class JsonStreamHandler(StreamHandler):

    __whitelist__ = (
        'asctime',
        'created',
        'filename',
        'funcName',
        'levelname',
        'levelno',
        'lineno',
        'message',
        'module',
        'msecs',
        'name',
        'pathname',
        'process',
        'processName',
        'relativeCreated',
        'thread',
        'threadName',
        'taskName',
    )

    def __init__(
            self,
            stream: Literal['stdout', 'stderr'] = 'stdout',
            field: str = 'message',
            *fields,
            **kwargs
    ):
        super().__init__(getattr(sys, self.__valid(stream)))
        self.fields = self.__permissible(field) + self.__permissible(fields)
        self.kwargs = kwargs

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

    def __permissible(self, fields: str | Iterable[str]) -> tuple[str, ...]:
        """Ensure that fields are white-listed strings or iterables thereof."""
        if isinstance(fields, str):
            return self.__whitelisted(fields.strip()),
        return tuple(self.__whitelisted(field.strip()) for field in fields)

    def __whitelisted(self, field: str) -> str:
        """Ensure that every requested field is on the whitelist."""
        if field in self.__whitelist__:
            return field
        else:
            tmp = '"{}" is not a valid log-record attribute'
            raise ValueError(tmp.format(field))

    @property
    def default(self) -> dict[str, Callable[[Any], str]]:
        """The fallback "default" for `json.dumps` is the builtin `str()`."""
        return {'default': str}

    def emit(self, record: LogRecord) -> None:
        """Process log records and forward to parent `emit()` method."""
        # Dictionary with requested fields plucked from the log record
        fields = {
            field: getattr(record, field)
            for field in self.fields
            if field not in ('asctime', 'message')  # These are handled later
        }
        # If requested, add the string-formatted timestamp
        if 'asctime' in self.fields:
            fields.update({'asctime': self.asctime(record.created)})
        # If requested, add the log message as nested field ...
        if 'message' in self.fields:
            merged = fields | {'message': record.msg}
        # ... or merge requested field the message if not.
        else:
            # If the message is indeed a dictionary, this should work ...
            try:
                merged = record.msg | fields
            # ... but if it does not, we fall back to a nested message.
            except TypeError:
                merged = fields | {'message': record.msg}

        # Try to serialize the final JSON message ...
        try:
            serialized = json.dumps(merged, **self.kwargs)
        # ... but fall back to a working "default" if it doesn't work.
        except TypeError:
            serialized = json.dumps(merged, **(self.kwargs | self.default))

        # Set the serialized json as log message of the record and emit.
        record.msg = serialized
        super().emit(record)

    @staticmethod
    def asctime(timestamp: float) -> str:
        """Format the logging timestamp to string, including milliseconds."""
        return datetime.fromtimestamp(
            timestamp
        ).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

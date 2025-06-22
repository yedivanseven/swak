import sys
import json
import logging
from typing import Any, Literal
from collections.abc import Iterable, Mapping
from functools import cached_property
from datetime import datetime
from logging import Logger, StreamHandler, Handler, LogRecord
from ..repr import ArgRepr
from .formats import JSON_FMT


class JsonStreamHandler(StreamHandler):
    """StreamHandler for logging JSON-formatted messages.

    Parameters
    ----------
    stream: str, optional
        Must be one of "stdout" or "stderr". Defaults to "stdout".
    field: str or iterable, optional
        A single `LogRecord attribute <https://docs.python.org/3/library/
        logging.html#logrecord-attributes>`_ or an iterable thereof. All values
        that are not marked by "You shouldn’t need to format this yourself."
        are allowed. Logged messages will be merged with a dictionary of these
        keys before being written to `stream` unless you explicitly refer to
        the LogRecord attribute "message", in which case that field will
        contain the dictionary you want to log.
        Defaults to ``("levelname", "name")``.
    *fields: str
        Additional LogRecord attributes to merge with the log message.
    **extras
        Additional, static key-value pairs to merge into every log message.

    Raises
    ------
    TypeError
        If `stream` is not a string.
    ValueError
        If `stream` is neither "stdout" nor "stderr" or if any of the given
        `fields` are not a permissible LogRecord attribute.

    """

    __whitelist__ = {
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
        'taskName'
    }

    def __init__(
            self,
            stream: Literal['stdout', 'stderr'] = 'stdout',
            field: str | Iterable[str] = JSON_FMT,
            *fields: str,
            **extras: Any
    ) -> None:
        super().__init__(stream=getattr(sys, self.__valid(stream)))
        self.fields = self.__permissible(field) | self.__permissible(fields)
        self.extras = extras

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

    def __permissible(self, fields: str | Iterable[str]) -> set[str]:
        """Ensure that fields are white-listed strings or iterables thereof."""
        if isinstance(fields, str):
            return {self.__whitelisted(fields.strip().lower())}
        return {self.__whitelisted(str(f).strip().lower()) for f in fields}

    def __whitelisted(self, field: str) -> str:
        """Ensure that every requested field is on the whitelist."""
        if field in self.__whitelist__:
            return field
        else:
            tmp = '"{}" is not a valid log-record attribute'
            raise ValueError(tmp.format(field))

    @property
    def basics(self) -> set[str]:
        """LogRecord attributes to add to the log message."""
        return self.fields - {'message'}

    def format(self, record: LogRecord) -> str:
        """Format the LogRecord as a JSON-serialized string."""
        # To simplify, add converted timestamp, even if we don't need it
        record.asctime = self.asctime(record.created)

        # Dictionary with requested fields plucked from the log record
        basics = {basic: getattr(record, basic) for basic in self.basics}

        # If requested, add the log message as a nested field ...
        if 'message' in self.fields:
            message = basics | {'message': record.msg}
        # ... or, if not, merge the requested fields with the message.
        else:
            # If the message is indeed a dictionary, this should work ...
            try:
                message = {**record.msg, **basics}
            # ... but if it does not, we fall back to a nested message.
            except TypeError:
                message = basics | {'message': record.msg}

        # Return the serialized JSON as the log message to be emitted.
        return json.dumps(
            self.extras | message,
            sort_keys=True,
            default=lambda x: x.as_json if hasattr(x, 'as_json') else repr(x),
        )

    @staticmethod
    def asctime(timestamp: float) -> str:
        """Format a unix timestamp to string, including milliseconds."""
        return datetime.fromtimestamp(timestamp).strftime(
            '%Y-%m-%d %H:%M:%S.%f'
        )[:-3]


class JsonLogger(ArgRepr):
    """Wrapped logger to stdout or stderr with a JSON-formatted StreamHandler.

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
        the LogRecord attribute "message", in which case that field will
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

    def log(self, level: int, message: Mapping[str, Any]) -> tuple[()]:
        """Log a message at the given level.

        Parameters
        ----------
        level: int
            The level to log at.
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        self.logger.log(level, message)
        return ()

    def debug(self, message: Mapping[str, Any]) -> tuple[()]:
        """Log a DEBUG message.

        Parameters
        ----------
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.DEBUG, message)

    def info(self, message: Mapping[str, Any]) -> tuple[()]:
        """Log an INFO message.

        Parameters
        ----------
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.INFO, message)

    def warning(self, message: Mapping[str, Any]) -> tuple[()]:
        """Log a WARNING message.

        Parameters
        ----------
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.WARNING, message)

    def error(self, message: Mapping[str, Any]) -> tuple[()]:
        """Log an ERROR message.

        Parameters
        ----------
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.ERROR, message)

    def critical(self, message: Mapping[str, Any]) -> tuple[()]:
        """Log a CRITICAL message.

        Parameters
        ----------
        message: Mapping
            The (dictionary) message to log.

        Returns
        -------
        tuple
            An empty tuple.

        """
        return self.log(logging.CRITICAL, message)

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

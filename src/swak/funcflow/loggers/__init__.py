"""Loggers specifically designed to keep your functions clean and tidy.

To avoid polluting your code, these loggers are meant to be inserted
into your functional workflow *between* its building blocks. Arguments passed
in are simply passed through to the next step, but you still have access to
them to customize the logged message.

"""

from .formats import DEFAULT_FMT, PID_FMT, RAW_FMT, SHORT_FMT, JSON_FMT
from .std import PassThroughStdLogger
from .file import PassThroughFileLogger
from .json import PassThroughJsonLogger

__all__ = [
    'PassThroughStdLogger',
    'PassThroughFileLogger',
    'PassThroughJsonLogger',
    'DEFAULT_FMT',
    'SHORT_FMT',
    'PID_FMT',
    'RAW_FMT',
    'JSON_FMT'
]

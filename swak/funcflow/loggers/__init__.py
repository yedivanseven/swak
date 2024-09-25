"""Loggers specifically designed to keep your functions clean and tidy.

To avoid polluting your code, these loggers are meant to be inserted
into your functional workflow *between* its building blocks. Arguments passed
in are simply passed through to the next step, but you still have access to
them to customize the logged message.

"""

from .stdout import PassThroughStdOut, DEFAULT_FMT, PID_FMT

__all__ = [
    'PassThroughStdOut',
    'DEFAULT_FMT',
    'PID_FMT'
]

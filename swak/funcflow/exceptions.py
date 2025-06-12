from typing import Any
from collections.abc import Callable


class FilterError(Exception):
    pass


class ForkError(Exception):
    pass


class MapError(Exception):
    pass


class PipeError(Exception):
    pass


class ReduceError(Exception):
    pass


class RouteError(Exception):
    pass


class SplitError(Exception):
    pass


class SumError(Exception):
    pass


class FallbackErrors(ExceptionGroup):
    pass


class SafeError(Exception):
    """Special exception to wrap other exceptions.

    Parameters
    ----------
    error: Exception
        The wrapped exception.
    name: str
        A human-readable string representation of the callable that failed.
    call: callable
        The callable that failed.
    call_args: tuple
        The arguments that `call` failed with.

    See Also
    --------
    Safe

    """

    def __init__(
            self,
            error: Exception,
            name: str,
            call: type | Callable[..., Any],  # Not sure if we really need this
            call_args: tuple[Any, ...]
    ) -> None:
        self.error = error
        self.name = name
        self.call = call
        self.call_args = call_args
        super().__init__(self.message)

    @property
    def message(self):
        """The error message displayed if the exception is actually raised."""
        msg = '\nCaught {} calling\n{}\nwith arguments\n{}\n{}'
        err_cls = self.error.__class__.__name__
        return msg.format(err_cls, self.name, self.call_args, self.error)

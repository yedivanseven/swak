import os
from typing import Any, overload


def exit_ok(*_: Any) -> int:
    """Accepts any number of arguments but always returns exit code for OK.

    Returns
    -------
    int
        Exit code for OK.

    """
    return os.EX_OK


def unit(*_: Any) -> tuple[()]:
    """Accepts any number of arguments but always returns and empty tuple.

    Returns
    -------
    tuple
        Empty tuple.

    """
    return ()


@overload
def identity() -> tuple[()]:
    ...


@overload
def identity[T](arg: T) -> T:
    ...


@overload
def identity[T, *Ts](arg: T, *args: *Ts) -> tuple[T, *Ts]:
    ...


def identity(*args):
    """Pass through any number of argument(s) doing nothing.

    Parameters
    ----------
    *args
        Argument(s) to pass through.

    Returns
    -------
    object or tuple
        Called with one argument, this argument is simply passed through. When
        called with more than one argument, the arguments tuple returned.

    """
    return args[0] if len(args) == 1 else args
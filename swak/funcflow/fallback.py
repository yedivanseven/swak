from typing import Any, Self
from collections.abc import Callable, Iterable, Iterator
from functools import singledispatchmethod
from ..misc import IndentRepr
from .misc import unit
from .exceptions import FallbackErrors


class Fallback[**P, T](IndentRepr):
    """Try different options in case a callable fails to process its input.

    Upon subclassing and/or instantiation, type annotation with the list
    of argument types that the callable(s) to try take as input, followed by
    the common return type(s) of the callable(s), is recommended.

    Parameters
    ----------
    calls: Callable or iterable
        A single callable or an iterable thereof with the same call signature
        and return value(s). These will be called, one after the other, with
        the input arguments to (callable) instances until no error occurs.
        Providing and empty iterable makes instances simply return their
        input argument(s).
    *errors: Exception class
        Exception classes that are caught and trigger trying the next of the
        `calls`. If none are given, all exceptions qualify.
    callback: Callable, optional
        Will be called each time one of the `calls` fails raising one of the
        `errors` with (a) the name of the failing callable, (b) a tuple of the
        arguments it was called with, and (c) the exception it raised.
        Defaults to `unit`, which  does nothing at all.

    Raises
    ------
    TypeError
        If `calls` is neither a callable nor an iterable thereof.

    """
    def __init__(
            self,
            calls: Callable[P, T] | Iterable[Callable[P, T]],
            *errors: type[Exception],
            callback: Callable[[str, P, Exception], Any] = unit
    ) -> None:
        self.calls = self.__valid(calls)
        self.errors = tuple(set(errors)) if errors else (Exception,)
        self.callback = callback
        super().__init__(self.calls,*self.errors, callback=callback)

    def __iter__(self) -> Iterator[Callable[P, T]]:
        # We could also iterate over instances of self ...
        return self.calls.__iter__()

    def __len__(self) -> int:
        return self.calls.__len__()

    def __bool__(self) -> bool:
        return bool(self.calls)

    def __contains__(self, item: Callable[P, T]) -> bool:
        return item in self.calls

    def __reversed__(self) -> NotImplemented:
        return NotImplemented

    @singledispatchmethod
    def __getitem__(self, index: int) -> Callable[P, T]:
        # We could also return instances of self ...
        return self.calls[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(
            self.calls[index],
            *self.errors,
            callback=self.callback
        )

    def __hash__(self) -> int:
        return hash((self.calls, self.errors, self.callback))

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return (
                self.calls == other.calls and
                self.errors == other.errors and
                self.callback is other.callback
            )
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return (
                self.calls != other.calls
                or self.errors != other.errors
                or self.callback is not other.callback
            )
        return NotImplemented

    def __add__(
            self,
            other: Callable[P, T] | Iterable[Callable[P, T]] | Self
    ) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(
                [*self.calls, *other.calls],
                *{*self.errors, *other.errors},
                callback=self.callback
            )
        try:
            return self.__class__(
                [*self.calls, *self.__valid(other)],
                *self.errors,
                callback=self.callback
            )
        except TypeError:
            return NotImplemented

    def __radd__(
            self,
            other: Callable[P, T] | Iterable[Callable[P, T]]
    ) -> Self:
        try:
            return self.__class__(
                [*self.__valid(other), *self.calls],
                *self.errors,
                callback=self.callback
            )
        except TypeError:
            return NotImplemented

    def __call__(self, *args: P.args) -> T:
        """Call the cached callables in turn until one does not raise errors.

        Parameters
        ----------
        *args
            Argument(s) to the cached callables.

        Returns
        -------
        object
            Whatever the cached callables return.

        """
        if self.calls:
            errors = []
            for call in self.calls:
                try:
                    args = call(*args)
                except self.errors as error:
                    errors.append(error)
                    self.callback(self._name(call), args, error)
                else:
                    break
            else:
                raise FallbackErrors('All options exhausted!', errors)
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args

    @staticmethod
    def __valid(
        calls: Callable[P, T] | Iterable[Callable[P, T]],
    ) -> tuple[Callable[P, T], ...]:
        """Ensure that the argument is indeed (an iterable of) callable(s)."""
        if callable(calls):
            return (calls,)
        iterable = True
        all_callable = False
        try:
            all_callable = all(callable(call) for call in calls)
        except TypeError:
            iterable = False
        if iterable and all_callable:
            return tuple(calls)
        raise TypeError('All calls must be callable!')

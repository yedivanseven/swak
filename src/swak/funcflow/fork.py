from typing import Any, Self, ParamSpec
from collections.abc import Iterator, Callable, Iterable
from functools import singledispatchmethod
from ..misc import IndentRepr
from .exceptions import ForkError

P = ParamSpec('P')
type Call = type | Callable[P, Any]


class Fork[**P, T](IndentRepr):
    """Call any number of callables with the same argument(s).

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that all callables take, followed by a ``tuple``
    specifying the concatenation of the return types of all callables, ignoring
    empty tuples. If only a single object remains, the type of that object
    should be annotated.

    Parameters
    ----------
    call: callable or iterable of callables, optional
        One callable or an iterator of callables to all call with the same
        arguments. Defaults to an empty tuple.
    *calls: callable
        Additional callables to call with the same argument(s).

    Raises
    ------
    ForkError
        If (any of) `call` or any of `calls` are not, in fact, callable.

    """

    def __init__(self, call: Call | Iterable[Call] = (), *calls: Call) -> None:
        self.calls = self.__valid(call) + self.__valid(calls)
        super().__init__(self.calls)

    def __iter__(self) -> Iterator[Call]:
        # We could also iterate over instances of self ...
        return self.calls.__iter__()

    def __len__(self) -> int:
        return self.calls.__len__()

    def __bool__(self) -> bool:
        return bool(self.calls)

    def __contains__(self, call: Call) -> bool:
        return call in self.calls

    def __reversed__(self) -> Self:
        return self.__class__(*reversed(self.calls))

    @singledispatchmethod
    def __getitem__(self, index: int) -> Call:
        # We could also return instances of self ...
        return self.calls[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(*self.calls[index])

    def __hash__(self) -> int:
        return self.calls.__hash__()

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.calls == other.calls
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.calls != other.calls
        return NotImplemented

    def __add__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.calls, *other.calls)
        try:
            return self.__class__(self.calls, *self.__valid(other))
        except ForkError:
            return NotImplemented

    def __radd__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(other.calls, *self.calls)
        try:
            return self.__class__(self.__valid(other), *self.calls)
        except ForkError:
            return NotImplemented

    def __call__(self, *args: P.args) -> T:
        """Call all specified `calls` with the same argument(s).

        Parameters
        ----------
        *args
            Arguments to call all `calls` with.

        Returns
        -------
        tuple or object
            Concatenation of all return values of all `calls` in order. If only
            one of the `calls` returns something other than an empty tuple,
            that object is returned.

        Raises
        ------
        ForkError
            When one of the `calls` raises an exception.

        """
        results = []
        for i, call in enumerate(self.calls):
            try:
                result = call(*args)
            except Exception as error:
                msg = '\n{} executing\n{}\nin fork {} of\n{}\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(err_cls, name, i, self, error)
                raise ForkError(fmt) from error
            else:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
        return results[0] if len(results) == 1 else tuple(results)

    @staticmethod
    def __valid(calls: Call | Iterable[Call]) -> tuple[Call, ...]:
        """Ensure that the argument is indeed an iterable of callables."""
        if callable(calls):
            return calls,
        iterable = True
        all_callable = False
        try:
            all_callable = all(callable(call) for call in calls)
        except TypeError:
            iterable = False
        if iterable and all_callable:
            return tuple(calls)
        raise ForkError('All branches in the fork must be callable!')

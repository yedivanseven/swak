from typing import Iterator, Any, Callable, Self, Iterable
from functools import singledispatchmethod
from .exceptions import ForkError
from ..magic import IndentRepr


class Fork[**P, T](IndentRepr):
    """Call any number of callables with the same argument(s).

    This class is instantiated with one or more callables. The (callable)
    object is then called with one or more arguments compatible with the
    call signature of all these callables. The return value(s) of all
    callables are concatenated to a tuple, ordered in the same order as
    the callables are given.

    Parameters
    ----------
    *calls: callable
        Callables to call with the same argument(s).

    Notes
    -----
    Upon instantiation, the generic class can be type-annotated with the
    Tuple of return types of all `calls`.

    """

    def __init__(self, *calls: Callable[P, Any]) -> None:
        super().__init__(*calls)
        self.calls = calls

    def __iter__(self) -> Iterator[Callable[P, Any]]:
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def __contains__(self, call: Callable[P, Any]) -> bool:
        return call in self.calls

    def __reversed__(self) -> Self:
        return self.__class__(*reversed(self.calls))

    @singledispatchmethod
    def __getitem__(self, index: int) -> Callable[P, Any]:
        return self.calls[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(*self.calls[index])

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, Fork):
            return self.calls == other.calls
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, Fork):
            return self.calls != other.calls
        return NotImplemented

    def __add__(
            self,
            other: Callable[P, Any] | Iterable[Callable[P, Any]] | Self
    ) -> Self:
        if isinstance(other, Fork):
            return self.__class__(*self.calls, *other.calls)
        try:
            _ = [callable(call) for call in other]
            return self.__class__(*self.calls, *other)
        except TypeError:
            try:
                _ = callable(other)
                return self.__class__(*self.calls, other)
            except TypeError:
                return NotImplemented

    def __radd__(
            self,
            other: Callable[P, Any] | Iterable[Callable[P, Any]] | Self
    ) -> Self:
        if isinstance(other, Fork):
            return self.__class__(*other.calls, *self.calls)
        try:
            _ = [callable(call) for call in other]
            return self.__class__(*other, *self.calls)
        except TypeError:
            try:
                _ = callable(other)
                return self.__class__(other, *self.calls)
            except TypeError:
                return NotImplemented

    def __call__(self, *args: P.args) -> T:
        """Call all specified callables with the same argument(s).

        Parameters
        ----------
        *args
            Arguments to call all `calls` with.

        Returns
        -------
        tuple
            Concatenation of all return values of all `calls` orders in the
            order of `calls`.

        Raises
        ------
        ForkError
            When one of the `calls` raises an exception.

        """
        results = []
        for i, call in enumerate(self):
            try:
                result = call(*args)
            except Exception as error:
                msg = 'Error executing\n{}\nin fork {} of\n{}\n{}:\n{}'
                name = error.__class__.__name__
                fmt = msg.format(self._name(call), i, self, name, error)
                raise ForkError(fmt)
            else:
                if isinstance(result, tuple):
                    results.extend(result)
                else:
                    results.append(result)
        n_res = len(results)
        return tuple(results) if n_res == 0 or n_res > 1 else results[0]

from typing import Iterator, Any, Callable, Self, Iterable, ParamSpec
from functools import singledispatchmethod
from ..magic import IndentRepr
from .exceptions import SideEffectsError


P = ParamSpec('P')
type Call = type | Callable[P, Any]


class SideEffects[**P](IndentRepr):
    """Call any number of callables with the same argument(s) and return those.

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that all callables take as input (and that will
    be returned when calling instances).

    Parameters
    ----------
    call: callable or iterable of callables, optional
        One callable or an iterator of callables to all call with the same
        arguments. Defaults to an empty tuple.
    *calls: callable
        Additional callables to call with the same argument(s).

    """

    def __init__(self, call: Call | Iterable[Call] = (), *calls: Call) -> None:
        self.calls = ((call,) if callable(call) else tuple(call)) + calls
        super().__init__(self.calls)

    def __iter__(self) -> Iterator[Call]:
        # We could also iterate over instances of self ...
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def __contains__(self, item: Call) -> bool:
        return item in self.calls

    def __reversed__(self) -> NotImplemented:
        return NotImplemented

    @singledispatchmethod
    def __getitem__(self, index: int) -> Call:
        # We could also return instances of self ...
        return self.calls[index]

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(*self.calls[index])

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, SideEffects):
            return self.calls == other.calls
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, SideEffects):
            return self.calls != other.calls
        return NotImplemented

    def __add__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, SideEffects):
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

    def __radd__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, SideEffects):
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

    def __call__(self, *args: P.args) -> P.args:
        """Call all specified `calls` with the same argument(s).

        Parameters
        ----------
        *args
            Arguments to call all `calls` with.

        Returns
        -------
        tuple or object
            The `args` that the instances was called with. If only a single
            argument was given, that object is returned.

        Raises
        ------
        SideEffectsError
            When one of the `calls` raises an exception.

        """
        for i, call in enumerate(self):
            try:
                call(*args)
            except Exception as error:
                msg = '\n{} executing\n{}\nin step {} of\n{}\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(err_cls, name, i, self, error)
                raise SideEffectsError(fmt)
        return args[0] if len(args) == 1 else args

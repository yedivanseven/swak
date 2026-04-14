from typing import Any, Self
from collections.abc import Iterator, Callable, Iterable
from functools import singledispatchmethod
from ..misc import IndentRepr
from .exceptions import PipeError

type Call = type | Callable[..., Any]


class Pipe[**P, T](IndentRepr):
    """Chain any number of callable objects into a single callable object.

    Arguments passed to the functional composition will be forwarded to the
    first callable in the chain. Subsequent callables will be called with
    the return value(s) of the previous callable in the chain. The return
    value of the functional composition is the return value of the last
    callable in the chain.

    Parameters
    ----------
    call: callable or iterable of callables, optional
        One callable or an iterator of callables to chain one after another.
        Defaults to an empty tuple.
    *calls: callable
        Additional callables to chain one after another.

    Raises
    ------
    PipeError
        If (any of) `call` or any of `calls` are not, in fact, callable.

    Note
    ----
    Upon instantiation, the generic class can be type-annotated with the list
    of argument types of the first callable in the chain, followed by the
    return type of the last callable.

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

    def __contains__(self, item: Call) -> bool:
        return item in self.calls

    def __reversed__(self):
        raise TypeError(f'{type(self).__name__} objects cannot be reversed')

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
        except PipeError:
            return NotImplemented

    def __radd__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(other.calls, *self.calls)
        try:
            return self.__class__(self.__valid(other), *self.calls)
        except PipeError:
            return NotImplemented

    def __call__(self, *args: P.args) -> T:
        """Call the functional composition this object was instantiated with.

        Parameters
        ----------
        *args
            Arguments to pass to the first callable in `calls`.

        Returns
        -------
        object
            Whatever the last callable of `calls` returns.

        Raises
        ------
        PipeError
            When one of the callables in the chain raises an exception.

        """
        for i, call in enumerate(self):
            try:
                args = call(*args) if isinstance(args, tuple) else call(args)
            except Exception as error:
                msg = '\n{} executing\n{}\nin step {} of\n{}\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(err_cls, name, i, self, error)
                raise PipeError(fmt) from error
        return  args[0] if isinstance(args, tuple) and len(args) == 1 else args

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
        raise PipeError('All items in the pipe must be callable!')

from typing import Iterator, Any, Callable, Self, Iterable
from functools import singledispatchmethod
from ..magic import IndentRepr
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

    Notes
    -----
    Upon instantiation, the generic class can be type-annotated with the list
    of argument types of the first callable in the chain, followed by the
    return type of the last callable.

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
        if isinstance(other, Pipe):
            return self.calls == other.calls
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, Pipe):
            return self.calls != other.calls
        return NotImplemented

    def __add__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, Pipe):
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
        if isinstance(other, Pipe):
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
        args = args[0] if len(args) == 1 else args
        for i, call in enumerate(self):
            try:
                args = call(*args) if isinstance(args, tuple) else call(args)
            except Exception as error:
                msg = '\n{} executing\n{}\nin step {} of\n{}\n{}'
                err_cls = error.__class__.__name__
                name = self._name(call)
                fmt = msg.format(err_cls, name, i, self, error)
                raise PipeError(fmt)
        return args

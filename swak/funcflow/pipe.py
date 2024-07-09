from typing import Iterator, Any, Callable, Self, Iterable, ParamSpec
from functools import singledispatchmethod
from .exceptions import PipeError
from ..magic import ArgRepr, IndentRepr

P = ParamSpec('P')
CallT = type | Callable[..., Any]
CallsT = tuple[CallT, ...]
SidesT = dict[int, tuple[int, ...]]
CacheT = dict[int, Any]


# ToDo: Allow side-loading kwargs!
class Side(ArgRepr):

    def __init__(self, *steps: int) -> None:
        super().__init__(steps)
        self.steps = steps


class Pipe[**P, T](IndentRepr):
    """Chain any number of callable objects into a single callable object.

    Arguments passed to the functional composition will be forwarded to the
    first callable in the chain. Subsequent callables will be called with
    the return value(s) of the previous callable in the chain. The return
    value of the functional composition is the return value of the last
    callable in the chain.

    Parameters
    ----------
    *calls: callable
        Callables to chain one after another.

    Notes
    -----
    Upon instantiation, the generic class can be type-annotated with the list
    of argument types of the first callable in the chain, followed by the
    return type of the last callable.

    """

    def __init__(self, *calls: CallT | Side) -> None:
        self.calls, self.__sides, self.__cache = self.__parse(*calls)
        super().__init__(*calls)

    @staticmethod
    def __parse(*calls: CallT) -> tuple[CallsT, SidesT, CacheT]:
        sides = {}
        callables = []
        n_callables = 0
        for item in calls:
            if isinstance(item, Side):
                sides.update({n_callables: item.steps})
            else:
                callables.append(item)
                n_callables += 1
        cache = {k: None for k in set(sum(sides.values(), ()))}
        return tuple(callables), sides, cache

    def __iter__(self) -> Iterator[CallT]:
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __bool__(self) -> bool:
        return self.__len__() > 0

    def __contains__(self, item: CallT) -> bool:
        return item in self.calls

    def __reversed__(self) -> NotImplemented:
        return NotImplemented

    @singledispatchmethod
    def __getitem__(self, index: int) -> CallT:
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

    def __add__(
            self,
            other: CallT | Iterable[CallT] | Self
    ) -> Self:
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

    def __radd__(
            self,
            other: CallT | Iterable[CallT] | Self
    ) -> Self:
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
        Whatever the last callable of `calls` returns.

        Raises
        ------
        PipeError
            When one of the callables in the chain raises an exception.

        """
        for i, call in enumerate(self):
            if i in self.__cache:
                self.__cache[i] = args if isinstance(args, tuple) else (args,)
            cached = sum([self.__cache[j] for j in self.__sides.get(i, ())], ())
            try:
                if isinstance(args, tuple):
                    args = call(*args, *cached)
                else:
                    args = call(args, *cached)
            except Exception as error:
                msg = 'Error executing\n{}\nin step {} of\n{}\n{}:\n{}'
                err_cls = error.__class__.__name__
                fmt = msg.format(self._name(call), i, self, err_cls, error)
                raise PipeError(fmt)
        return args

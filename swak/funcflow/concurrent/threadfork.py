from typing import Iterator, Any, Callable, Self, Iterable, ParamSpec
from functools import singledispatchmethod
from concurrent.futures import ThreadPoolExecutor
from ...magic import IndentRepr
from ..exceptions import ForkError

P = ParamSpec('P')
type Call = type | Callable[P, Any]


class ThreadFork[**P, T](IndentRepr):
    """Thread-parallely call any number of callables with the same argument(s).

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
    max_workers: int, optional
        Maximum number of worker threads used in the pool to call
        `calls` asynchronously. Will be forwarded to the constructor of
        ``ThreadPoolExecutor``. Defaults to 16.
    thread_name_prefix: str, optional
        Will be forwarded to the constructor of ``ThreadPoolExecutor``.
        Defaults to an empty string.
    initializer: callable, optional
        Called at the start of each worker thread. Will be forwarded to the
        constructor of ``ThreadPoolExecutor``. Defaults to ``None``.
    initargs: tuple, optional
        Arguments passed to the initializer. Will be forwarded to the
        constructor of ``ThreadPoolExecutor``. Defaults to an empty tuple.
    timeout: int or float, optional
        Maximum time (in seconds) to wait for results to be available. Defaults
        to ``None``, which means there is no limit for the time to wait. Will
        be forwarded to the ``result`` method of the ``Future``.

    See Also
    --------
    concurrent.futures.ThreadPoolExecutor

    """

    def __init__(
            self,
            call: Call | Iterable[Call] = (),
            *calls: Call,
            max_workers: int = 16,
            thread_name_prefix: str = '',
            initializer: Callable[..., Any] | None = None,
            initargs: tuple[Any, ...] = (),
            timeout: int | float | None = None
    ) -> None:
        self.calls = ((call,) if callable(call) else tuple(call)) + calls
        super().__init__(*self.calls)
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.initializer = initializer
        self.initargs = initargs
        self.timeout = timeout

    def _indented_repr(self, level: int) -> str:
        cls = self.__class__.__name__
        args = ', '.join([
            str(self.max_workers),
            repr(self.thread_name_prefix),
            self._name(self.initializer),
            str(self.initargs),
            str(self.timeout)
        ])
        body = '\n'.join(super()._indented_repr(level).split('\n')[1:])
        head = f'{cls}({args})' + (':' if body else '')
        return head + ('\n' if body else '') + body

    def __iter__(self) -> Iterator[Call]:
        # We could also iterate over instances of self ...
        return iter(self.calls)

    def __len__(self) -> int:
        return len(self.calls)

    def __bool__(self) -> bool:
        return self.__len__() > 0

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

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, ThreadFork):
            return self.calls == other.calls
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, ThreadFork):
            return self.calls != other.calls
        return NotImplemented

    def __add__(self, other: Call | Iterable[Call] | Self) -> Self:
        if isinstance(other, ThreadFork):
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
        if isinstance(other, ThreadFork):
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
        """Concurrently call all specified `calls` with the same argument(s).

        Parameters
        ----------
        *args
            Arguments to call all `calls` with.

        Returns
        -------
        tuple or object
            Concatenation of all return values of all `calls` in order. If only
            one of the `calls` returns something other than an empty tuple, that
            object is returned.

        Raises
        ------
        ForkError
            When one of the `calls` raises an exception.

        """
        with ThreadPoolExecutor(
                self.max_workers,
                self.thread_name_prefix,
                self.initializer,
                self.initargs
        ) as pool:
            futures = [pool.submit(call, *args) for call in self.calls]
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(self.timeout)
                except Exception as error:
                    msg = '\n{} executing\n{}\nin fork {} of\n{}\n{}'
                    err_cls = error.__class__.__name__
                    name = self._name(self.calls[i])
                    fmt = msg.format(err_cls, name, i, self, error)
                    raise ForkError(fmt)
                else:
                    if isinstance(result, tuple):
                        results.extend(result)
                    else:
                        results.append(result)
        return results[0] if len(results) == 1 else tuple(results)

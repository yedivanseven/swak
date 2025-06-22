from typing import Any
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from ...misc import ArgRepr
from ..exceptions import MapError


# ToDo: Add a "flatten" option that weeds out emtpy tuples from the results
class ThreadMap[**P, S, T](ArgRepr):
    """Partial of ``concurrent.futures.ThreadPoolExecutor.map``.

    Upon subclassing and/or instantiation, type annotation with a list of the
    argument type(s) of `transform`, the return type of `call`, and the return
    type of `wrapper` is recommended.

    Parameters
    ----------
    transform: callable
        Transforms element(s) of the input iterable(s).
    wrapper: type or callable, optional
        If not given, an attempt will be made to return the type of the first
        iterable the callable instance is being called with (by calling its
        class with a list of the mapped elements). If explicitly given,
        `wrapper` will be called with a list of mapped elements. Consequently,
        the return type will be the (return) type of `wrapper`.
    max_workers: int, optional
        Maximum number of worker threads used in the pool to execute
        `transform` asynchronously. Will be forwarded to the constructor of
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
        be forwarded to the ``map`` method of the ``ThreadPoolExecutor``.

    Note
    ----
    In contrast to calling the ``map`` method of a ``ThreadPoolExecutor``
    directly, which returns a generator object, the mapped iterable is fully
    manifested first and only then wrapped.

    See Also
    --------
    concurrent.futures.ThreadPoolExecutor

    """

    def __init__(
            self,
            transform: type[S] | Callable[P, S],
            wrapper: type[T] | Callable[[list[S]], T] | None = None,
            max_workers: int = 16,
            thread_name_prefix: str = '',
            initializer: Callable[..., Any] | None = None,
            initargs: tuple[Any, ...] = (),
            timeout: float | None = None
    ) -> None:
        super().__init__(
            transform,
            wrapper,
            max_workers,
            thread_name_prefix,
            initializer,
            initargs,
            timeout
        )
        self.transform = transform
        self.wrapper = wrapper
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.initializer = initializer
        self.initargs = initargs
        self.timeout = timeout

    def __call__(self, iterable: Iterable, *iterables: Iterable) -> T:
        """Concurrently transform the element(s) of the given iterable(s).

        Parameters
        ----------
        iterable: Iterable
            An iterable of elements to transform.
        *iterables: Iterable
            If given, the cached `transform` is called with the corresponding
            elements of `ìterable` and all `iterables` as arguments.

        Returns
        -------
        Sequence
            Same type as `iterable` if `wrapper` was not specified on
            instantiation or the (return) type of `wrapper`. Note that, as with
            python's builtin ``map`` function, the length of the output
            sequence is limited by the shortest of the input iterables.

        Raises
        ------
        MapError
            If calling the ``ThreadPoolExecutor``'s ``map`` method raises an
            exception or if wrapping the results leads to an exception.

        """
        with ThreadPoolExecutor(
            self.max_workers,
            self.thread_name_prefix,
            self.initializer,
            self.initargs
        ) as pool:
            mapped = pool.map(
                self.transform,
                iterable,
                *iterables,
                timeout=self.timeout
            )
            try:
                mapped = list(mapped)
            except Exception as error:
                msg = ('Error calling\n{}\non one or more '
                       'element(s) of the iterable(s)!\n{}:\n{}')
                name = self._name(self.transform)
                err_cls = error.__class__.__name__
                fmt = msg.format(name, err_cls, error)
                raise MapError(fmt) from error
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            wrapped = wrap(mapped)
        except Exception as error:
            msg = '\n{} calling wrapper\n{}\non map results:\n{}'
            name = self._name(wrap)
            err_cls = error.__class__.__name__
            raise MapError(msg.format(err_cls, name, error)) from error
        return wrapped

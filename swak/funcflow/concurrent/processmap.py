from typing import Any
from collections.abc import Iterable, Callable
from concurrent.futures import ProcessPoolExecutor
from ...misc import ArgRepr
from ..exceptions import MapError


# ToDo: Add a "flatten" option that weeds out emtpy tuples from the results
class ProcessMap[**P, S, T](ArgRepr):
    """Partial of ``concurrent.futures.ProcessPoolExecutor.map``.

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
        Maximum number of worker processes used in the pool to execute
        `transform` asynchronously. Will be forwarded to the constructor of
        ``ProcessPoolExecutor``. Defaults to 4.
    initializer: callable, optional
        Called at the start of each worker process. Will be forwarded to the
        constructor of ``ProcessPoolExecutor``. Defaults to ``None``.
    initargs: tuple, optional
        Arguments passed to the initializer. Will be forwarded to the
        constructor of ``ProcessPoolExecutor``. Defaults to an empty tuple.
    max_tasks_per_child: int, optional
        Maximum number of iterable items to transform in each worker process
        before they are being restarted. Defaults to ``None`` indicating no
        restart(s) at all.
    timeout: int or float, optional
        Maximum time (in seconds) to wait for results to be available. Defaults
        to ``None``, which means there is no limit for the time to wait. Will
        be forwarded to the ``map`` method of the ``ProcessPoolExecutor``.
    chunksize: int, optional
        Number of items from the iterable to feed to one worker process at a
        time. Defaults to 1. Will be forwarded to the ``map`` method of the
        ``ProcessPoolExecutor``.

    Note
    ----
    In contrast to calling the ``map`` method of a ``ProcessPoolExecutor``
    directly, which returns a generator object, the mapped iterable is fully
    manifested first and only then wrapped.

    See Also
    --------
    concurrent.futures.ProcessPoolExecutor

    """

    def __init__(
            self,
            transform: type[S] | Callable[P, S],
            wrapper: type[T] | Callable[[list[S]], T] | None = None,
            max_workers: int | None = 4,
            initializer: Callable[..., Any] | None = None,
            initargs: tuple[Any, ...] = (),
            max_tasks_per_child: int | None = None,
            timeout: float | None = None,
            chunksize: int = 1
    ) -> None:
        super().__init__(
            transform,
            wrapper,
            max_workers,
            initializer,
            initargs,
            max_tasks_per_child,
            timeout,
            chunksize
        )
        self.transform = transform
        self.wrapper = wrapper
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs
        self.max_tasks_per_child = max_tasks_per_child
        self.timeout = timeout
        self.chunksize = chunksize

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
            If calling the ``ProcessPoolExecutor``'s ``map`` method raises an
            exception or if wrapping the results leads to an exception.

        """
        with ProcessPoolExecutor(
                self.max_workers,
                None,
                self.initializer,
                self.initargs,
                max_tasks_per_child=self.max_tasks_per_child
        ) as pool:
            mapped = pool.map(
                self.transform,
                iterable,
                *iterables,
                timeout=self.timeout,
                chunksize=self.chunksize
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

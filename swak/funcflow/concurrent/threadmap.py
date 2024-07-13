from typing import Any, Iterable, Callable
from concurrent.futures import ThreadPoolExecutor
from ..exceptions import MapError
from ...magic import ArgRepr


class ThreadMap[**P, S, T](ArgRepr):
    """Equivalent to a partial of the python builtin ``map`` function.

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
        `wrapper` will be called with a list mapped elements. Consequently,
        the return type will be the (return) type of `wrapper`.

    Notes
    -----
    In contrast to python's builtin lazy ``map`` function, which returns a
    generator object, the mapped iterable is fully manifested first and only
    then wrapped.

    """

    def __init__(
            self,
            transform: type[S] | Callable[P, S],
            wrapper: type[T] | Callable[[list[S]], T] | None = None,
            max_workers: int = 16,
            thread_name_prefix: str = '',
            initializer: Callable[..., Any] | None = None,
            initargs: tuple[Any, ...] = (),
            timeout: int | float | None = None,
    ) -> None:
        super().__init__(transform, wrapper)
        self.transform = transform
        self.wrapper = wrapper
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.initializer = initializer
        self.initargs = initargs,
        self.timeout = timeout

    def __call__(self, iterable: Iterable, *iterables: Iterable) -> T:
        """Transform the element(s) of the given iterable(s).

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
            If calling the cached `transform` on any element(s) of the given
            iterable(s) raises an exception or if wrapping the results leads
            to an exception.

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
                raise MapError(fmt)
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            wrapped = wrap(mapped)
        except Exception:
            msg = 'Could not wrap map results into an instance of {}!'
            raise MapError(msg.format(self._name(wrap)))
        return wrapped

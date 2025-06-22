from collections.abc import Iterable, Callable
from ..misc import ArgRepr
from .exceptions import MapError


# ToDo: Add a "flatten" option that weeds out emtpy tuples from the results
class Map[**P, S, T](ArgRepr):
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

    Note
    ----
    In contrast to python's builtin lazy ``map`` function, which returns a
    generator object, the mapped iterable is fully manifested first and only
    then wrapped.

    """

    def __init__(
            self,
            transform: type[S] | Callable[P, S],
            wrapper: type[T] | Callable[[list[S]], T] | None = None
    ) -> None:
        super().__init__(transform, wrapper)
        self.transform = transform
        self.wrapper = wrapper

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
        mapped = []
        for i, elements in enumerate(zip(*(iterable, *iterables))):
            try:
                mapped.append(self.transform(*elements))
            except Exception as error:
                msg = '\n{} calling\n{}\non element #{}:\n{}\n{}'
                name = self._name(self.transform)
                err_cls = error.__class__.__name__
                display = elements[0] if len(elements) == 1 else elements
                fmt = msg.format(err_cls, name, i, display, error)
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

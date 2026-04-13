from collections.abc import Iterable, Callable
from ..misc import ArgRepr
from .exceptions import MapError


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
    flat: bool, optional
        If ``True``, tuple return values of `transform` are unpacked into the
        output sequence rather than kept as a single element. Non-tuple return
        values are left untouched, regardless of this flag.
        Defaults to ``False``.

    Note
    ----
    In contrast to python's builtin lazy ``map`` function, which returns a
    generator object, the mapped iterable is fully manifested first and only
    then wrapped.

    Important
    ---------
    If `flat` is ``True``, the length of the output sequence may differ from
    the length of the input sequence, as tuple return values are spliced into
    the output rather than appended as a single element.

    """

    def __init__(
            self,
            transform: type[S] | Callable[P, S],
            wrapper: type[T] | Callable[[list[S]], T] | None = None,
            flat: bool = False
    ) -> None:
        super().__init__(transform, wrapper, flat)
        self.transform = transform
        self.wrapper = wrapper
        self.flat = flat

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
                result = self.transform(*elements)
            except Exception as error:
                msg = '\n{} calling\n{}\non element #{}:\n{}'
                name = self._name(self.transform)
                err_cls = error.__class__.__name__
                fmt = msg.format(err_cls, name, i, error)
                raise MapError(fmt) from error
            if self.flat and isinstance(result, tuple):
                mapped.extend(result)
            else:
                mapped.append(result)
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            wrapped = wrap(mapped)
        except Exception as error:
            msg = '\n{} calling wrapper\n{}\non map results:\n{}'
            name = self._name(wrap)
            err_cls = error.__class__.__name__
            raise MapError(msg.format(err_cls, name, error)) from error
        return wrapped

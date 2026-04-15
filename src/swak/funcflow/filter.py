from collections.abc import Callable, Iterable
from ..misc import ArgRepr
from .exceptions import FilterError


class Filter[S, T](ArgRepr):
    """Equivalent to a partial of the python builtin ``filter`` function.

    Upon subclassing and/or instantiation, type annotation with the type of the
    elements in the iterator to be filtered and the return type of `wrapper`
    is recommended.

    Parameters
    ----------
    criterion: callable, optional
        Callable accepting one element of an iterable at a time and returning
        a boolean value for each one of them. Defaults to ``None``, which
        returns the inherent truth values of the object in the iterable.
    wrapper: type or callable, optional
        If not given, an attempt will be made to return the same type of
        iterable the callable instance is being called with (by calling its
        class with a list of the filtered elements). If explicitly given,
        `wrapper` will be called with a list of filtered elements.
        Consequently, the return type will be the (return) type of `wrapper`.

    Note
    ----
    In contrast to python's builtin lazy ``filter`` function, which returns a
    generator object, the filtered iterable is fully manifested first and only
    then wrapped.

    """

    def __init__(
            self,
            criterion: Callable[[S], bool] | None = None,
            wrapper: type[T] | Callable[[list[S]], T] | None = None
    ) -> None:
        super().__init__(criterion, wrapper)
        self.criterion = criterion
        self.wrapper = wrapper

    def __call__(self, iterable: Iterable[S]) -> T:
        """Filter an iterable according to the specified criterion.

        Parameters
        ----------
        iterable: Iterable
            An iterable whose elements are to be filtered according to whether
            the `criterion` evaluates to ``True`` or ``False``.

        Returns
        -------
        Sequence
            Same type as `iterable` if `wrapper` was not specified on
            instantiation or the (return) type of `wrapper`.

        Raises
        ------
        FilterError
            If calling the criterion on any element of `iterable` raises
            an exception or if wrapping the results leads to an exception.

        """
        criterion = bool if self.criterion is None else self.criterion
        filtered = []
        for i, element in enumerate(iterable):
            try:
                criterion_is_fulfilled = criterion(element)
            except Exception as error:
                msg = '\n{} calling criterion\n{}\non element #{}:\n{}\n{}'
                name = self._name(criterion)
                err_cls = error.__class__.__name__
                fmt = msg.format(err_cls, name, i, element, error)
                raise FilterError(fmt) from error
            if criterion_is_fulfilled:
                filtered.append(element)
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            wrapped = wrap(filtered)
        except Exception as err:
            msg = '\n{} calling wrapper\n{}\non filter results:\n{}'
            name = self._name(wrap)
            err_cls = err.__class__.__name__
            raise FilterError(msg.format(err_cls, name, err)) from err
        return wrapped

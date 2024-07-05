from typing import Callable, Iterable, Any
from .exceptions import FilterError
from ..magic import ArgRepr


class Filter[T](ArgRepr):
    """Partial of the python builtin ``filter`` function.

    Generic type annotation with the (return) type of `wrapper` is recommended.

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
        `wrapper` will be called with the list filtered elements. Consequently,
        the return type will the (return) type of `wrapper`.

    Notes
    -----
    In contrast to python's builtin lazy ``filter`` function, the filtered
    iterable is fully manifested first here and only then wrapped.

    """

    def __init__(
            self,
            criterion: Callable[[Any], bool] | None = None,
            wrapper: type[T] | Callable[[list], T] | None = None
    ) -> None:
        super().__init__(criterion, wrapper)
        self.criterion = criterion
        self.wrapper = wrapper

    def __call__(self, iterable: Iterable) -> T:
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
        try:
            filtered = list(filter(self.criterion, iterable))
        except Exception:
            msg = 'Could not call {} on all elements of the sequence!'
            raise FilterError(msg.format(self._name(self.criterion)))
        wrap = iterable.__class__ if self.wrapper is None else self.wrapper
        try:
            wrapped = wrap(filtered)
        except Exception:
            msg = 'Could not wrap the results into a instance of {}!'
            raise FilterError(msg.format(self._name(wrap)))
        return wrapped

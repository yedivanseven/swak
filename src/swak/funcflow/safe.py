from collections.abc import Callable, Iterable
from ..misc import ArgRepr
from .exceptions import SafeError


class Safe[**P, T](ArgRepr):
    """Wrap callable to catch potential errors and safely return them.

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that the callable takes, followed by the return
    type of the callable.

    Parameters
    ----------
    call: callable
        Callable to wrap.
    error: optional
        Specific exception class to catch (or an iterable thereof).
        Defaults to ``Exception``.
    *errors
        Additional exception classes to catch.

    See Also
    --------
    SafeError

    """

    def __init__(
            self,
            call: type[T] | Callable[P, T],
            error: type[Exception] | Iterable[type[Exception]] = (),
            *errors: type[Exception]
    ) -> None:
        self.call = call
        self.errors = self.__actual(error) + self.__actual(errors)
        super().__init__(call, *self.errors)
        self.errors = self.errors or (Exception,)

    def __call__(self, *args: P.args) -> T | SafeError:
        """Call the cached callable, catching any or all exceptions raised.

        Parameters
        ----------
        *args
            Arguments to call `call` with.

        Returns
        -------
        object or SafeError
            Either the return value(s) of the cached `call` or an instance of
            ``SafeError`` wrapping one of the specified `exceptions`.

        """
        try:
            return self.call(*args)
        except self.errors as error:
            return SafeError(error, self._name(self.call), self.call, args)

    @staticmethod
    def __actual(
            errors: type[Exception] | Iterable[type[Exception]]
    ) -> tuple[type[Exception], ...]:
        """Ensure that the provided errors are actually, well, errors."""
        if isinstance(errors, type) and issubclass(errors, Exception):
            return errors,
        iterable = True
        all_exceptions = False
        try:
            all_exceptions = all(
                isinstance(error, type) and issubclass(error, Exception)
                for error in errors
            )
        except TypeError:
            iterable = False
        if iterable and all_exceptions:
            return tuple(set(errors))
        raise TypeError('All errors must derive from Exception!')

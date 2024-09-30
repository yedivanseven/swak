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
    exception: optional
        Specific exception to catch (or an iterable of exceptions).
        Defaults to ``Exception``.
    *exceptions
        Additional exceptions to catch.

    See Also
    --------
    SafeError

    """

    def __init__(
            self,
            call: type[T] | Callable[P, T],
            exception: type[Exception] | Iterable[type[Exception]] = (),
            *exceptions: type[Exception]
    ) -> None:
        self.call = call
        try:
            exception = tuple(exception)
        except TypeError:
            exception = exception,
        self.exceptions = tuple(set(exception + exceptions))
        super().__init__(call, *self.exceptions)
        self.exceptions = self.exceptions if self.exceptions else (Exception,)

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
        except self.exceptions as error:
            return SafeError(error, self._name(self.call), self.call, args)

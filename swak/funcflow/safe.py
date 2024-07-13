from typing import Callable
from ..magic import ArgRepr


# ToDo: Add optional arg exception (single exception or tuple thereof)
class Safe[**P, T](ArgRepr):
    """Wrap callable to catch any errors it might raise and return them instead.

    Generic type annotation of instances is recommended. Provide a list of
    one or more input types that the callable takes, followed by the return type
    of the callable.

    Parameters
    ----------
    call: callable
        Callable to wrap.
    *exceptions
        Specific Exceptions to catch. If none are given all subclasses of
        ``Exception`` will be caught and returned indiscriminately.

    """

    def __init__(
            self,
            call: type[T] | Callable[P, T],
            *exceptions: type[Exception]
    ) -> None:
        super().__init__(call, *exceptions)
        self.call = call
        self.exceptions = exceptions if exceptions else (Exception, )

    def __call__(self, *args: P.args) -> T | Exception:
        """Call the cached callable, catching any or all exceptions raised.

        Parameters
        ----------
        *args
            Arguments to call `call` with.

        Returns
        -------
        object or Exception
            Either the return value(s) of the cached `call` or any of the
            specified `exceptions` should `call` raise any.

        """
        try:
            return self.call(*args)
        except self.exceptions as error:
            return error

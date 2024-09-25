from typing import Any
from collections.abc import Callable
from ..misc import ArgRepr


class Partial[T](ArgRepr):
    """Alternative implementation of ``functools.partial``.

    Positional arguments given when instances are called are *appended* to
    those given at instantiation . Upon subclassing and/or instantiation,
    annotation with the (return) type of `call` is recommended.

    Parameters
    ----------
    call: callable
        Callable object with arbitrary call signature.
    *args
        Any number of arguments to call `call` with.
    **kwargs
        Any number of keyword arguments to call `call` with.

    See Also
    --------
    Curry

    """

    def __init__(
            self,
            call: type[T] | Callable[..., T],
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(call, *args, **kwargs)
        self.call = call
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        """Execute cached callable with combined args and kwargs.

        Parameters
        ----------
        *args
            Additional arguments are appended to cached arguments.
        **kwargs
            Additional keyword arguments are combined with cached keyword
            arguments, overriding them in case of conflict.

        Returns
        -------
        object
            Whatever `call` returns.

        """
        return self.call(*(*self.args, *args), **{**self.kwargs, **kwargs})

from collections.abc import Callable
from ...misc import ArgRepr


class Delayed[T, **P](ArgRepr):
    """Recursively delay instantiating heavy classes until actually needed.

    To avoid blocking CPU/GPU memory with instances of huge PyTorch models
    until they are actually needed, both the model class and the required
    (keyword) arguments are cached until instances of this wrapper are called
    with *no* (keyword) arguments. At that point, classes are called and,
    thus, instantiated with the cached (keyword) arguments and, if any of
    these are wrapped themselves, they will be recursively called as well.

    Parameters
    ----------
    call: type or callable
        Class or function or method to be called with the cached (keyword)
        arguments once this wrapper is called.
    *args
        Optional arguments that `call` will be called with. Any `args` that
        are instances of this wrapper themselves will be called first and their
        return value will be used as argument to `call` instead.
    **kwargs
        Optional keyword arguments that `call` will be called with.
        Any `kwargs` that are instances of this wrapper themselves will be
        called first and their return value will be used as keyword argument
        to `call` instead.

    """

    def __init__(
            self,
            call: type[T] | Callable[P, T],
            *args: P.args,
            **kwargs: P.kwargs
    ) -> None:
        super().__init__(call)
        self.call = call
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *_, **__) -> T:
        """Call `call` with cached keyword (arguments) recursively resolved."""
        return self.call(
            *[
                arg() if isinstance(arg, self.__class__) else arg
                for arg in self.args
            ],
            **{
                key: value() if isinstance(value, self.__class__) else value
                for key, value in self.kwargs.items()
            }
        )

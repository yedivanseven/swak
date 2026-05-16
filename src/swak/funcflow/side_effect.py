from collections.abc import Callable
from ..misc import ArgRepr
from .exceptions import SideEffectError


class SideEffect[*Ts](ArgRepr):
    """Route call arguments into and past a side effect.

    Instances of this class are callable with any number of arguments
    (including no arguments), but must be called with exact number (and
    types) that the wrapped side effect expects. Arguments will be forwarded
    to the side effect and routed past it to be returned. The types of the
    arguments can be made explicit by type annotating the class.

    Parameters
    ----------
    call : callable
        A callable that accepts one or more arguments and returns nothing.

    """

    def __init__(self, call: Callable[[*Ts], None]) -> None:
        super().__init__(call)
        self.call = call

    def __call__(self, *args: *Ts) -> tuple[*Ts]:
        """Call the cached side effect and return its call arguments.

        Parameters
        ----------
        *args
            The arguments that the wrapped `call` expects.

        Returns
        -------
        object or tuple
            Empty tuple if called with no arguments, the single call argument
            if called with just one, the tuple of call arguments when called
            with more than one.

        """
        try:
            self.call(*args)
        except Exception as error:
            msg = '\n{} calling side effect\n{}\n{}'
            name = self._name(self.call)
            err_cls = error.__class__.__name__
            raise SideEffectError(msg.format(err_cls, name, error))
        return args[0] if len(args) == 1 else args

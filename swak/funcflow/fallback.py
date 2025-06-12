from collections.abc import Callable, Iterable
from ..misc import IndentRepr
from .exceptions import FallbackErrors


# ToDo: Add unit tests!
class Fallback[**P, T](IndentRepr):
    def __init__(
        self,
        calls: Callable[P, T] | Iterable[Callable[P, T]],
        *errors: Exception,
        callback,
    ) -> None:
        self.calls = self.__valid(calls)
        self.errors = errors or (Exception,)
        self.callback = callback
        super().__init__(self.calls,*self.errors, callback=callback)

    def __call__(self, *args: P.args) -> T:
        if self.calls:
            errors = []
            for call in self.calls:
                try:
                    args = call(*args)
                except self.errors as error:
                    errors.append(error)
                    self.callback(self._name(call), args, error)
                else:
                    break
            else:
                raise FallbackErrors('All options exhausted!', errors)
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args

    @staticmethod
    def __valid(
        calls: Callable[P, T] | Iterable[Callable[P, T]],
    ) -> tuple[Callable[P, T], ...]:
        """Ensure that the argument is indeed (an iterable of) callable(s)."""
        if callable(calls):
            return (calls,)
        iterable = True
        all_callable = False
        try:
            all_callable = all(callable(call) for call in calls)
        except TypeError:
            iterable = False
        if iterable and all_callable:
            return tuple(calls)
        raise TypeError('All calls must be callable!')

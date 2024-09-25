from collections.abc import Iterable, Callable
from ..misc import ArgRepr
from .exceptions import ReduceError


class Reduce[T, S](ArgRepr):
    """Equivalent to a partial of the ``functools.reduce`` function.

    Upon subclassing and/or instantiation, type annotation with the return type
    of `call` (which must be the same as the type of its first argument as well
    as the type of the accumulator `acc`) and the type ot its second argument
    (which must be the same as the type of the elements in the iterable acted
    upon) is recommended.

    Parameters
    ----------
    call: callable
        Callable accepting the current `acc` and the next element of iterable,
        returning the updated value of `acc`.
    acc: optional
        Initial value for the accumulator in the reduce operation. If not given
        calling reduce on an empty iterable will fail. Defaults to ``None``.

    """

    def __init__(
            self,
            call: Callable[[T, S], T],
            acc: T | None = None
    ) -> None:
        super().__init__(call, acc)
        self.call = call
        self.acc = acc

    def __call__(self, iterable: Iterable[S]) -> T:
        """Reduce an iterable by accumulating elements with the cached `call`.

        Parameters
        ----------
        iterable
            An iterable of elements to reduce.

        Returns
        -------
        object
            The reduced iterable.

        """
        iterator = iter(iterable)
        acc = next(iterator) if self.acc is None else self.acc
        for i, element in enumerate(iterator):
            try:
                acc = self.call(acc, element)
            except Exception as error:
                msg = '\n{} calling\n{}\non element #{}:\n{}\n{}'
                name = self._name(self.call)
                err_cls = error.__class__.__name__
                fmt = msg.format(err_cls, name, i + 1, element, error)
                raise ReduceError(fmt) from error
        return acc

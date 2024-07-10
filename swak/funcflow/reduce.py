from typing import Iterable, Callable
from ..magic import ArgRepr
from .exceptions import ReduceError


class Reduce[T, S](ArgRepr):
    """Partial of the ``functools`` module ``reduce`` function.

    This class is instantiated with a function that takes two arguments
    (an accumulator of type T and an element of type S) and returns a single
    object of type T. An initial value for the accumulator can also be
    provided. The (callable) object is then called on a sequence with elements
    of type S, returning a single object of type T.

    Parameters
    ----------
    call: callable
        Callable accepting two arguments, returning one.
    acc: optional
        Initial value for the accumulator in the reduce operation.

    Notes
    -----
    Upon instantiation, the generic class can be type-annotated with two
    types, the first for the type of the elements in the sequence to
    reduce, the second with the type of the accumulator, i.e., the return
    value of the reduction.

    """

    def __init__(self, call: Callable[[T, S], T], acc: T | None = None) -> None:
        super().__init__(call, acc)
        self.call = call
        self.acc = acc

    def __call__(self, iterable: Iterable[S]) -> T:
        """Reduce a sequence by accumulating elements with the specified call.

        Parameters
        ----------
        iterable
            A sequence of elements to reduce.

        Returns
        -------
        The reduced sequence.

        """
        iterator = iter(iterable)
        acc = next(iterator) if self.acc is None else self.acc
        for i, element in enumerate(iterator):
            try:
                acc = self.call(acc, element)
            except Exception as error:
                msg = 'Error calling\n{}\non element #{}:\n{}\n{}:\n{}'
                name = self._short(self.call)
                err_cls = error.__class__.__name__
                fmt = msg.format(name, i + 1, element, err_cls, error)
                raise ReduceError(fmt)
        return acc

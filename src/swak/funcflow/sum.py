from collections.abc import Iterable
from ..misc import ArgRepr
from .exceptions import SumError


class Sum[S, T](ArgRepr):
    """Equivalent to a partial of the python builtin ``sum`` function.

    Upon subclassing and/or instantiation, type annotation with the type of
    the elements in the iterable and the type of the final sum is recommended.

    Parameters
    ----------
    acc: optional
        The initial value of the sum to which all other elements are added.
        If not given, calling sum on an empty iterable will fail.
        Defaults to ``None``.

    """

    def __init__(self, acc: T | None = None) -> None:
        super().__init__(acc)
        self.acc = acc

    def __call__(self, iterable: Iterable[S]) -> T:
        """Sum up the elements of an iterable.

        Parameters
        ----------
        iterable: Iterable
            An iterable of elements to sum up.

        Returns
        -------
        object
            The sum of the elements in the `iterable`.

        """
        iterator = iter(iterable)
        acc = next(iterator) if self.acc is None else self.acc
        offset = 1 if self.acc is None else 0
        for i, element in enumerate(iterator):
            try:
                acc = acc + element
            except Exception as error:
                msg = 'Error adding element #{}:\n{}\n{}:\n{}'
                err_cls = error.__class__.__name__
                fmt = msg.format(i + offset, element, err_cls, error)
                raise SumError(fmt) from error
        return acc

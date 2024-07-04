from operator import itemgetter
from typing import Hashable, Mapping, Callable
from ..magic import ArgRepr


class ValuesGetter[T](ArgRepr):
    """Extract the sequence of dictionary values according to the given `keys`.

    Generic type annotation with the (return) type of `wrapper` is recommended.

    Parameters
    ----------
    *keys: hashable
        Arbitrary number of dictionary keys.
    wrapper: callable, optional
        A type or some other callable that transforms a tuple of dictionary
        values into some other type of sequence. The type of the sequence
        should be annotated. Instances can be type annotated with the type
        of sequence to be returned. Defaults to ``list``.

    """

    def __init__(
            self,
            *keys: Hashable,
            wrapper: type[T] | Callable[[tuple], T] = list
    ) -> None:
        super().__init__(*keys, wrapper=wrapper)
        self.keys = keys
        self.wrapper = wrapper

    @property
    def n_items(self) -> int:
        """Number of values to extract from dictionary."""
        return len(self.keys)

    def __call__(self, mapping: Mapping) -> T:
        """Extract sequence of dictionary values for given keys.

        Parameters
        ----------
        mapping: Mapping
            Dictionary to extract values from.

        Returns
        -------
        Sequence
            The values extracted from the dictionary. The type of sequence
            is determined by the return type of `wrap`. Defaults to ``list``.

        """
        match self.n_items:
            case 0:
                values = tuple()
            case 1:
                values = itemgetter(*self.keys)(mapping),
            case _:
                values = itemgetter(*self.keys)(mapping)
        return self.wrapper(values)

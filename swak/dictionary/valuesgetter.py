from operator import itemgetter
from typing import Hashable, Mapping, Callable
from ..magic import ArgRepr


class ValuesGetter[T](ArgRepr):
    """Extract the sequence of dictionary values according to the given `keys`.

    Instances can be type annotated with the return type of `wrap`.

    Parameters
    ----------
    *keys: hashable
        Arbitrary number of dictionary keys.
    wrap: callable, optional
        A type or some other callable that transforms a tuple of dictionary
        values into some other type of sequence. The type of the sequence
        should be annotated. Instances can be type annotated with the type
        of sequence to be returned. Defaults to ``list``.

    """

    def __init__(
            self,
            *keys: Hashable,
            wrap: type[T] | Callable[[tuple], T] = list
    ) -> None:
        super().__init__(*keys, wrap=wrap)
        self.keys = keys
        self.wrap = wrap

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
        return self.wrap(values)

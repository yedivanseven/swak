from operator import itemgetter
from typing import Hashable, Mapping, Callable, Self, Iterator, Iterable
from functools import singledispatchmethod
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

    def __len__(self) -> int:
        return len(self.keys)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __reversed__(self) -> Self:
        return self.__class__(*reversed(self.keys))

    def __iter__(self) -> Iterator[Self]:
        return iter(self.__class__(key) for key in self.keys)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.keys

    @singledispatchmethod
    def __getitem__(self, index: int) -> Self:
        return self.__class__(self.keys[index])

    @__getitem__.register
    def _(self, index: slice) -> Self:
        return self.__class__(*self.keys[index])

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, ValuesGetter):
            return self.keys == other.keys
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, ValuesGetter):
            return self.keys != other.keys
        return NotImplemented

    def __add__(self, other: Hashable | Iterable[Hashable] | Self) -> Self:
        if isinstance(other, ValuesGetter):
            return self.__class__(*self.keys, *other.keys)
        if isinstance(other, str):
            return self.__class__(*self.keys, other)
        try:
            _ = [hash(key) for key in other]
            return self.__class__(*self.keys, *other)
        except TypeError:
            try:
                _ = hash(other)
                return self.__class__(*self.keys, other)
            except TypeError:
                return NotImplemented

    def __radd__(self, other: Hashable | Iterable[Hashable] | Self) -> Self:
        if isinstance(other, ValuesGetter):
            return self.__class__(*other.keys, *self.keys)
        if isinstance(other, str):
            return self.__class__(other, *self.keys)
        try:
            _ = [hash(key) for key in other]
            return self.__class__(*other, *self.keys)
        except TypeError:
            try:
                _ = hash(other)
                return self.__class__(other, *self.keys)
            except TypeError:
                return NotImplemented

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
        match len(self):
            case 0:
                values = tuple()
            case 1:
                values = itemgetter(*self.keys)(mapping),
            case _:
                values = itemgetter(*self.keys)(mapping)
        return self.wrapper(values)

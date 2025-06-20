from operator import itemgetter
from typing import Self
from collections.abc import Hashable, Mapping, Callable, Iterator, Iterable
from functools import singledispatchmethod
from ..misc import ArgRepr


class ValuesGetter[T](ArgRepr):
    """Extract a sequence of dictionary values according to the given `keys`.

    Instances behave like sequences of these `keys` and can be interacted with
    accortdingly (e.g., one can index, slice, or reverse instances).

    Parameters
    ----------
    key: Hashable, optional
        Dictionary key or iterable thereof. Defaults to an empy tuple.
    *keys: Hashable
        Additional dictionary keys.
    wrapper: callable, optional
        A type or some other callable that transforms a tuple of dictionary
        values into some other type of sequence. The type of the sequence
        should be annotated. Instances can be type annotated with the type
        of sequence to be returned. Defaults to ``list``.

    Raises
    ------
    TypeError
        If (any of) `key` or any of `keys` are not, in fact, hashable.

    Note
    ----
    Generic type annotation with the (return) type of `wrapper` is recommended.

    Examples
    --------
    >>> get_values = ValuesGetter('name', 'age')
    >>> get_values({'name': 'John', 'weight': 84, 'age': 42})
    ['John', 42]

    """

    def __init__(
            self,
            key: Hashable | Iterable[Hashable] = (),
            *keys: Hashable,
            wrapper: type[T] | Callable[[tuple], T] = list
    ) -> None:
        key = self.__valid(key)
        self.keys: tuple[Hashable, ...] = key + self.__valid(keys)
        self.wrapper = wrapper
        super().__init__(*self.keys, wrapper=wrapper)

    def __len__(self) -> int:
        return self.keys.__len__()

    def __bool__(self) -> bool:
        return bool(self.keys)

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

    def __hash__(self) -> int:
        return self.keys.__hash__()

    def __eq__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.keys == other.keys
        return NotImplemented

    def __ne__(self, other: Self) -> bool:
        if isinstance(other, self.__class__):
            return self.keys != other.keys
        return NotImplemented

    def __add__(self, other: Hashable | Iterable[Hashable] | Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.keys, *other.keys)
        try:
            others = self.__valid(other)
        except TypeError:
            return NotImplemented
        return self.__class__(self.keys, *others)

    def __radd__(self, other: Hashable | Iterable[Hashable] | Self) -> Self:
        if isinstance(other, self.__class__):
            return self.__class__(other.keys, *self.keys)
        try:
            others = self.__valid(other)
        except TypeError:
            return NotImplemented
        return self.__class__(others, *self.keys)

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
                values = ()
            case 1:
                values = itemgetter(*self.keys)(mapping),
            case _:
                values = itemgetter(*self.keys)(mapping)
        return self.wrapper(values)

    @staticmethod
    def __valid(keys: Hashable | Iterable[Hashable]) -> tuple[Hashable, ...]:
        """Ensure that the argument is indeed an iterable of hashables."""
        if isinstance(keys, str):
            return keys,
        try:
            _ = [hash(key) for key in keys]
        except TypeError:
            _ = hash(keys)
            return keys,
        return tuple(keys)

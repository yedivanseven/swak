from typing import overload
from collections.abc import Mapping
from swak.misc import ArgRepr


class NoneDropper[T](ArgRepr):
    """Drop all entries with ``None`` values from a dictionary-like mapping.

    Parameters
    ----------
    mapping: Mapping, optional
        For convenience, the mapping to drop fields with ``None`` values from
        can already be given at instantiation, but nothing will happen until
        instances are actually called. Defaults to ``None``.

    Examples
    --------
    >>> drop = NoneDropper({'name': 'John', 'weight': 84, 'age': None})
    NoneDropperNoneDroppe>>> drop()
    {'name': 'John', 'weight': 82}

    >>> drop = NoneDropper()
    >>> drop({'name': 'John', 'weight': None, 'age': 42})
    {'name': 'John', 'age': 42}

    """

    def __init__(self, mapping: Mapping | None = None) -> None:
        super().__init__()
        self.mapping = mapping

    def __call__(self, mapping: Mapping | None = None) -> dict:
        """Drop entries with ``None`` values from a dictionary-like mapping.

        Parameters
        ----------
        mapping: Mapping, optional
            If no mapping was given on instantiation, one must be given here.
            Otherwise, there would be nothing to drop ``None`` values from.
            If a mapping was given on instantiation and one is given here, the
            latter replaces the former. Defaults to ``None``.

        Returns
        -------
        dict
            The dictionary given at instantiation or when calling the instance
            with all ``None`` value fields removed.

        Raises
        ------
        TypeError
            If the object to process, whether given on instantiation or when
            calling instances, is not dictionary-like.

        """
        obj = self.mapping if mapping is None else mapping
        if isinstance(obj, Mapping):
            return self.recursive(obj)
        raise TypeError('Input must be a dictionary-like mapping!')

    @overload
    def recursive(self, obj: Mapping) -> dict:
        ...

    @overload
    def recursive(self, obj: T) -> T:
        ...

    def recursive(self, obj):
        """Recursively drop fields with ``None`` values from a nested mapping.

        Parameters
        ----------
        obj
            Any python object.

        Returns
        -------
        object
            The object passed in (if it was not dictionary-like) or a
            dictionary with none of its values being ``None`` (if it was).

        """
        if not isinstance(obj, Mapping):
            return obj
        return {
            key: self.recursive(value)
            for key, value in obj.items()
            if value is not None
        }

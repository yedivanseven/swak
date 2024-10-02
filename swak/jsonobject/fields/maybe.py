from typing import Any
from collections.abc import Callable


class Maybe[T]:
    """Type annotation to allow for ``None`` values.

    Initialize this class with a built-in type, class, or other callable that
    will cast a given value to the desired type if it is not ``None``. Then
    calling the callable instance on that value will return the desired type
    if the value is not ``None`` and ``None`` if it is.

    Parameters
    ----------
    cast: callable
        Casts value to desired type if value is not ``None``.

    Note
    ----
    Upon instantiation, the generic class can be type-annotated with the
    return type of `cast`.

    """

    def __init__(self, cast: type[T] | Callable[[Any], T]) -> None:
        self.cast = cast

    def __call__(self, obj: Any) -> T:
        """Casts value to specified type if value is not ``None``.

        Parameters
        ----------
        obj
            Object to type-cast to, or ``None``.

        Returns
        -------
        object
            Type-cast object or ``None``.

        """
        obj_is_none_str = obj in ('null', 'None')
        return None if obj is None or obj_is_none_str else self.cast(obj)

from typing import Any
from collections.abc import Callable, Sequence


class ReprName:
    """Auxiliary mixin providing representations for (callable) objects."""

    def _repr(self, obj: Any, _: int = 0) -> str:
        """Representation for any object."""
        return self._name(obj) if callable(obj) else repr(obj)

    @staticmethod
    def _name(obj: None | type | Callable) -> str:
        """Representation for callable objects used for, e.g., error messages.

        Parameters
        ----------
        obj: callable
            Any callable object, e.g., a class, function, method, or lambda

        Returns
        -------
        str
            A human-readable representation of the callable `obj`.

        """
        if obj is None:
            return 'None'
        if isinstance(obj, ReprName):
            return repr(obj)
        try:
            name = obj.__qualname__
        except AttributeError:
            name = obj.__class__.__name__ + '(...)'
        return 'lambda' if '<lambda>' in name else name


class IndentRepr(ReprName):
    """Base class for a representation with numbered and indented children.

    This class is not meant to be instantiated by itself. Rather, it is meant
    to be inherited from. Its constructor is then meant to be called in the
    child's constructor (with ``super().__init__(...)``), passing as arguments
    all items that are desired to appear in a (zero-based) numbered-list
    representation. If any of these objects have inherited from ``IndentRepr``
    themselves, their representation will recursively be indented by another
    level. Additional (keyword) arguments will be included in the first line
    of the representation.

    Parameters
    ----------
    items: sequence, optional
        Objects to represent in a (zero-based) numbered list below the class.
        Defaults to an emtpy tuple.
    *args
        Additional arguments to appear in the class instantiation signature
        in the first line of the representation.
    **kwargs
        Additional keyword arguments to appear in the class instantiation
        signature in the first line of the representation.

    Examples
    --------
    >>> class Fancy(IndentRepr):
    ...     def __init__(self, foo, *args, **kwargs) -> None:
    ...         super().__init__(args, foo, **kwargs)
    ...
    >>> Fancy('bar', 1, 2, Fancy('baz', 3, 4), 5, answer=42)
    Fancy('bar', answer=42):
    [ 0] 1
    [ 1] 2
    [ 2] Fancy('baz'):
         [ 0] 3
         [ 1] 4
    [ 3] 5

    """

    def __init__(
            self,
            items: Sequence = (),
            *args: Any,
            **kwargs: Any
    ) -> None:
        self.__items = items
        self.__args = args
        self.__kwargs = kwargs

    def __repr__(self) -> str:
        return self._indented_repr(0)

    def _repr(self, obj: Any, level: int = 0) -> str:
        """Return (potentially) indented object representations."""
        if isinstance(obj, IndentRepr):
            return obj._indented_repr(level + 1)
        return super()._repr(obj)

    def _indented_repr(self, level: int) -> str:
        """Construct indented object representation."""
        cls = self.__class__.__name__
        args = ', '.join(self._repr(arg) for arg in self.__args)
        kwargs = (f'{k}={self._repr(v)}' for k, v in self.__kwargs.items())
        kwargs = ', '.join(kwargs)
        signature = ', '.join(filter(None, [args, kwargs]))
        suffix = ":\n" if self.__items else ""
        indent = 5 * level * ' '
        items = enumerate(self._repr(item, level) for item in self.__items)
        items = '\n'.join(f'{indent}[{i:>2}] {item}' for i, item in items)
        return f'{cls}({signature})' + suffix + items


class ArgRepr(ReprName):
    """Base class for a representation with class name and (keyword) arguments.

    This class is not meant to be instantiated by itself. Rather, it is meant
    to be inherited from. Its constructor is then meant to be called in the
    child's constructor (with ``super().__init__(...)``), passing the (keyword)
    arguments that are desired to appear in the child's representation.

    Parameters
    ----------
    *args
        Appear as arguments within parentheses in the child representation
        after its class name.
    **kwargs
        Appear as keyword arguments within parentheses in the child
        representation after its class name and arguments.

    Examples
    --------
    >>> class Pretty(ArgRepr):
    ...     def __init__(self, foo, answer):
    ...         super().__init__(foo, answer=answer)
    ...
    >>> Pretty('good', 42)
    Pretty('good', answer=42)

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__args = args
        self.__kwargs = kwargs

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = ', '.join(self._repr(arg) for arg in self.__args)
        kwargs = (f'{k}={self._repr(v)}' for k, v in self.__kwargs.items())
        kwargs = ', '.join(kwargs)
        signature = ', '.join(filter(None, [args, kwargs]))
        return f'{cls}({signature})'

    def _repr(self, obj: Any, _: int = 0) -> str:
        """Representation for any object."""
        if isinstance(obj, IndentRepr):
            head = repr(obj).splitlines()[0]
            head = head.removesuffix(':')
            suffix = f'[{len(repr(obj).splitlines()) - 1}]'
            return head + suffix
        return super()._repr(obj)

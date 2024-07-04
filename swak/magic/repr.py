from typing import Any


class _ReprName:
    """Auxiliary Mixin providing object representation logic."""

    def _repr(self, obj: Any, _: int = 0) -> str:
        """Representation for any object."""
        if callable(obj):
            return self._name(obj)
        return repr(obj)

    @staticmethod
    def _name(obj: Any) -> str:
        """Representation for callable objects."""
        if isinstance(obj, _ReprName):
            return repr(obj)
        try:
            name = obj.__qualname__
        except AttributeError:
            name = obj.__class__.__name__ + '(...)'
        return 'lambda' if '<lambda>' in name else name


class ArgRepr(_ReprName):
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


class IndentRepr(_ReprName):
    """Base class for a representation with numbered and indented children.

    This class is not meant to be instantiated by itself. Rather, it is meant
    to be inherited from. Its constructor is then meant to be called in the
    child's constructor (with ``super().__init__(...)```), passing as arguments
    all objects that are desired to appear in a (zero-based) numbered-list
    representation. If any of these objects have inherited from ``IndentRepr``
    themselves, their representation will recursively be indented by another
    level.

    Parameters
    ----------
    *args
        Objects to represent in a (zero-based) numbered list below the class.

    """

    def __init__(self, *args: Any) -> None:
        self.args = args

    def __repr__(self) -> str:
        return self._indented_repr(0)

    def _repr(self, item: Any, level: int = 0) -> str:
        """Return (potentially) indented object representations."""
        if isinstance(item, IndentRepr):
            return item._indented_repr(level + 1)
        return super()._repr(item)

    def _indented_repr(self, level: int) -> str:
        """Construct indented object representation."""
        cls = f'{self.__class__.__name__}:\n'
        indent = 5 * level * ' '
        args = enumerate(self._repr(arg, level) for arg in self.args)
        items = '\n'.join(f'{indent}[{i:>2}] {arg}' for i, arg in args)
        return cls + items

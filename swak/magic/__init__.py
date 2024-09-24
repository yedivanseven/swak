"""Base classes providing convenient implementations of common `magic` methods.

Examples
--------
>>> class Without:
...     def __init__(self, a: int) -> None:
...         self.a = a
...
>>> Without(42)
<__main__.Without at 0x7b15eb37ca70>

>>> class With(ArgRepr):
...     def __init__(self, a: int) -> None:
...         super().__init__(a)
...         self.a = a
...
>>> With(42)
With(42)

>>> class Fancy(IndentRepr):
...     def __init__(self, *args) -> None:
...         super().__init__(args)
...
>>> Fancy(1, 2, Fancy('a', 'b'), 3)
Fancy():
[ 0] 1
[ 1] 2
[ 2] Fancy():
     [ 0] 'a'
     [ 1] 'b'
[ 3] 3
"""

from .repr import ArgRepr, IndentRepr

__all__ = [
    'ArgRepr',
    'IndentRepr'
]

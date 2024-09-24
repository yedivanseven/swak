"""Convenient partials of functions for manipulating dictionaries.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dictionaries can flow through a preconfigured processing
pipe of callable objects.

Examples
--------
>>> get_values = ValuesGetter('name', 'age')
>>> get_values({'name': 'John', 'weight': 82, 'age': 32})
['John', 32]

"""

from .valuesgetter import ValuesGetter

__all__ = [
    'ValuesGetter'
]

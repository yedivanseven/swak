"""Tools for composing functions and other callable objects into workflows.

Once you have broken down your workflow into small, modular steps, the tools in
this package allow you to parameterize them with everything that is known at
program start, and compose them into arbitrarily complex execution graphs that
are easy to maintain and to extend.

Important
---------
In order for everything to work smoothly, there is only one thing you will have
to keep in mind when writing the composable building blocks of your workflow.
If a function (or other callable object) is to return *nothing*, that is, when
it is a *side effect*, don't just omit the ``return`` statement, but explicitly
return an *empty tuple*!

"""

from .filter import Filter
from .fork import Fork
from .map import Map
from .partial import Partial
from .curry import Curry
from .pipe import Pipe
from .reduce import Reduce
from .route import Route
from .split import Split
from .sum import Sum
from .fallback import Fallback
from .safe import Safe
from .exceptions import SafeError
from .misc import apply, unit, identity

__all__ = [
    'Filter',
    'Fork',
    'Map',
    'Partial',
    'Curry',
    'Pipe',
    'Reduce',
    'Route',
    'Split',
    'Sum',
    'Fallback',
    'Safe',
    'SafeError',
    'apply',
    'unit',
    'identity'
]

# ToDo: Add Retry
# ToDo: Add lazy sub-package

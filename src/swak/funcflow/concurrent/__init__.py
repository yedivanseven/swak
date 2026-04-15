"""Thread- and process-parallel versions for elements of your workflow."""

from .threadmap import ThreadMap
from .threadfork import ThreadFork
from .processmap import ProcessMap
from .processfork import ProcessFork

__all__ = [
    'ThreadMap',
    'ThreadFork',
    'ProcessMap',
    'ProcessFork'
]

# ToDo: Add ThreadRoute and ProcessRoute
# ToDo: Add lazy subpackage

"""Polars utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .misc import FromPandas
from .groupby import GroupByAgg
from .frame import (
    Select,
    Filter,
    Drop,
    Sort,
    WithColumns,
    GroupBy,
    GroupByDynamic,
    Join,
    ToPandas
)

__all__ = [
    'FromPandas',
    'Select',
    'Filter',
    'Drop',
    'Sort',
    'WithColumns',
    'GroupBy',
    'GroupByDynamic',
    'Join',
    'ToPandas',
    'GroupByAgg'
]

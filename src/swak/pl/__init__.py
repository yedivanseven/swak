"""Polars utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .misc import FromPandas, Create, Concat
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
    ToPandas,
    Cast,
    Rename,
    DropNulls,
    Pivot,
    VStack,
    Unique,
    Head,
    Tail
)

__all__ = [
    'Create',
    'FromPandas',
    'Concat',
    'Select',
    'Filter',
    'Drop',
    'Sort',
    'WithColumns',
    'GroupBy',
    'GroupByDynamic',
    'Join',
    'ToPandas',
    'GroupByAgg',
    'Cast',
    'Rename',
    'DropNulls',
    'Pivot',
    'VStack',
    'Unique',
    'Head',
    'Tail'
]

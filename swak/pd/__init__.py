"""Pandas utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .read import ParquetReader
from .write import ParquetWriter
from .groupby import GroupByApply, RollingGroupByApply
from .frame import (
    AsType,
    ColumnSelector,
    ColumnsSelector,
    ColumnMapper,
    RowsSelector,
    GroupBy,
    Join,
    Assign,
    Drop,
    DropNA,
    SortValues,
    SetIndex,
    ResetIndex,
    Rename,
    Agg,
    RollingWindow
)

__all__ = [
    'ParquetReader',
    'ParquetWriter',
    'AsType',
    'ColumnSelector',
    'ColumnsSelector',
    'ColumnMapper',
    'RowsSelector',
    'Join',
    'Assign',
    'Drop',
    'DropNA',
    'SortValues',
    'SetIndex',
    'ResetIndex',
    'Rename',
    'GroupBy',
    'Agg',
    'RollingWindow',
    'GroupByApply',
    'RollingGroupByApply',
]

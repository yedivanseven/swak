"""Pandas utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .read import ParquetReader
from .write import ParquetWriter
from .groupby import FrameGroupByAgg
from .frame import (
    AsType,
    ColumnSelector,
    ColumnsSelector,
    ColumnMapper,
    RowsSelector,
    FrameGroupBy,
    Join,
    Assign
)

__all__ = [
    'ParquetReader',
    'ParquetWriter',
    'AsType',
    'ColumnSelector',
    'ColumnsSelector',
    'ColumnMapper',
    'RowsSelector',
    'FrameGroupBy',
    'FrameGroupByAgg',
    'Join',
    'Assign'
]

# ToDo: Replicate for pola-rs in separate "po" subpackage!

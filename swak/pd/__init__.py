"""Pandas utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .read import ParquetReader
from .frame import (
    AsType,
    ColumnSelector,
    ColumnsSelector,
    ColumnMapper,
    RowsSelector
)

__all__ = [
    'ParquetReader',
    'AsType',
    'ColumnSelector',
    'ColumnsSelector',
    'ColumnMapper',
    'RowsSelector'
]

"""Pandas utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .read import ParquetReader
from .write import ParquetWriter
from .frame import (
    AsType,
    ColumnSelector,
    ColumnsSelector,
    ColumnMapper,
    RowsSelector
)

__all__ = [
    'ParquetReader',
    'ParquetWriter',
    'AsType',
    'ColumnSelector',
    'ColumnsSelector',
    'ColumnMapper',
    'RowsSelector'
]

# ToDo: Replicate for pola-rs in separate "po" subpackage!

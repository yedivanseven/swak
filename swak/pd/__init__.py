"""Pandas utilities and partials of dataframe method calls.

Parameters that are known at program start are used to initialize the classes
so that, at runtime, dataframes can flow through a preconfigured processing
pipe of callable objects.

"""

from .astype import AsType
from .column_selector import ColumnSelector
from .columns_selector import ColumnsSelector
from .mapper import Mapper
from .rows_selector import RowsSelector
from .groupby import GroupBy, GroupByApply, RollingGroupByApply
from .join import Join
from .assign import Assign
from .drop import Drop
from .dropna import DropNA
from .sort_values import SortValues
from .set_index import SetIndex
from .reset_index import ResetIndex
from .rename import Rename
from .agg import Agg
from .rolling_window import RollingWindow

__all__ = [
    'AsType',
    'ColumnSelector',
    'ColumnsSelector',
    'Mapper',
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

# ToDo: Add AsFreq, Explode, Transform, Copy, and FillNA
# ToDo: Search for "inplace" keywords and pop them!
# ToDo: Explore use of dispatch method more!
# ToDo: Don't forget methods of Rolling!

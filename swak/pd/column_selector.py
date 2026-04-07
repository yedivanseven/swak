from typing import overload
from collections.abc import Hashable
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series
from ..misc import ArgRepr


class ColumnSelector(ArgRepr):
    """Select a single column of a (grouped) pandas dataframe as a series.

    This is simply a partial for calling a (grouped) dataframe's
    ``__getitem__`` method with a single argument (using the square-brackets
    accessor).

    Parameters
    ----------
    col: hashable
        Single DataFrame column to select.

    """

    def __init__(self, col: Hashable) -> None:
        self.col = self.__valid(col)
        super().__init__(col)

    @overload
    def __call__(self, df: DataFrame) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> SeriesGroupBy:
        ...

    def __call__(self, df):
        """Select a single column of a (grouped) pandas dataframe as series.

        Parameters
        ----------
        df: DataFrame or DataFrameGroupBy
            Pandas dataframe or grouped dataframe to select column from.

        Returns
        -------
        Series or SeriesGroupBy
            The selected column from the (grouped) dataframe.

        """
        return df[self.col]

    @staticmethod
    def __valid(col: Hashable) -> Hashable:
        _ = hash(col)
        return col

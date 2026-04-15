from typing import overload
from collections.abc import Callable
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from ..misc import ArgRepr

type Mask = list[bool] | Series | ndarray[tuple[int], np.dtype[np.bool_]]
type Condition = Callable[[DataFrame | Series], Mask]


class RowsSelector(ArgRepr):
    """Select rows from a pandas dataframe or series with some condition.

    This is simply a partial for calling a dataframe's or series'
    ``__getitem__`` method (using the square-brackets accessor) with a
    callable that takes the dataframe or series as input, and produces a
    1-D, boolean array-like structure (of the same length as the dataframe
    or series to select from).

    Parameters
    ----------
    condition: callable or array-like
        A callable that accepts a dataframe or series and produces a 1-D,
        boolean array-like structure of the same length

    """

    def __init__(self, condition: Condition) -> None:
        super().__init__(condition)
        self.condition = condition

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df) :
        """Select rows from a pandas dataframe or series.

        Parameters
        ----------
        df: DataFrame or Series
            The pandas dataframe or series to select rows from.

        Returns
        -------
        DataFrame or Series
            The pandas dataframe or series with only the selected rows.

        """
        return df[self.condition]

from typing import Any, overload
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Axis


class FillNA(ArgRepr):
    """Light wrapper around a pandas dataframe or series ``fillna`` method.

    Parameters
    ----------
    value: scalar, dict, Series, or DataFrame
        Value to use to fill holes (e.g. 0), alternately a dict, Series, or
        DataFrame of values specifying which value to use for each index
        (for a Series) or column (for a DataFrame). Values not in the dict or
        Series or DataFrame will not be filled. This value cannot be a list.
    axis: int or str, optional
        Axis along which to fill missing values in case of a dataframe. Must
        be one of 0, "index", 1, or "columns". Ignored for Series.
        Defaults to 0.
    limit: int, optional
        This is the maximum number of entries along the entire axis where
        NaNs will be filled. Must be greater than 0 if not None.

    """

    def __init__(
            self,
            value: Any,
            axis: Axis = 0,
            limit: int | None = None
    ) -> None:
        self.value = value
        self.axis = axis
        self.limit = limit
        super().__init__(value, axis=axis, limit=limit)

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Call the ``fillna`` method of the passed pandas object.

        Parameters
        ----------
        df: Series or DataFrame
            The pandas object to call ``fillna`` on with the cached
            (keyword) arguments.

        Returns
        -------
        Series or DataFrame
            The same type as called with.

        """
        return df.fillna(
            self.value,
            axis=self.axis,
            inplace=False,
            limit=self.limit
        )

from typing import overload
from collections.abc import Hashable, Sequence
from pandas.core.groupby import DataFrameGroupBy
from pandas import DataFrame
from ..misc import ArgRepr


class ColumnsSelector(ArgRepr):
    """Select one or more columns of a (grouped) pandas dataframe as dataframe.

    This is simply a partial for calling a (grouped) dataframe's
    ``__getitem__``  method with a list of arguments (using the square-brackets
    accessor).

    Parameters
    ----------
    col: Hashable, optional
        Column name or sequence thereof. Defaults to an empty tuple.
    *cols: Hashable
        Additional columns names.

    """

    def __init__(
            self,
            col: Hashable | Sequence[Hashable] = (),
            *cols: Hashable
    ) -> None:
        col: tuple[Hashable, ...] = self.__valid(col)
        self.cols: tuple[Hashable, ...] = col + self.__valid(cols)
        super().__init__(*self.cols)

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> DataFrameGroupBy:
        ...

    def __call__(self, df):
        """Select the specified column(s) from a (grouped) pandas dataframe.

        Parameters
        ----------
        df: DataFrame or DataFrameGroupBy
            Pandas dataframe or grouped dataframe to select column(s) from.

        Returns
        -------
        DataFrame or DataFrameGroupBy
            The selected column(s) of the (grouped) dataframe.

        """
        return df[list(self.cols)]

    @staticmethod
    def __valid(cols: Hashable | Sequence[Hashable]) -> tuple[Hashable, ...]:
        """Ensure that the columns are indeed a sequence of hashables."""
        if isinstance(cols, str):
            return cols,
        try:
            _ = [hash(col) for col in cols]
        except TypeError:
            _ = hash(cols)
            return cols,
        return tuple(cols)

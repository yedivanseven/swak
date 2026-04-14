from typing import overload
from collections.abc import Hashable
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Labels


class Explode(ArgRepr):
    """Partial of a pandas dataframe or series ``explode`` method.

    Parameters
    ----------
    col: hashable or sequence, optional
        Column name or sequence of column names to explode. Only relevant
        when called on a DataFrame.
    *cols: hashable
        Additional column names to explode.
    ignore_index: bool, optional
        If ``True``, the resulting index will be reset. Otherwise, it will
        be exploded as well, introducing duplicates. Defaults to ``False``.

    """

    def __init__(
            self,
            col: Labels = (),
            *cols: Hashable,
            ignore_index: bool = False
    ) -> None:
        self.cols = self.__valid(col) + self.__valid(cols)
        self.ignore_index = ignore_index
        super().__init__(*self.cols, ignore_index=ignore_index)

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Explode a dataframe or series.

        Parameters
        ----------
        df: DataFrame or Series
            Pandas dataframe or series to explode.

        Returns
        -------
        DataFrame or Series
            Exploded pandas dataframe or series.

        Raises
        ------
        TypeError
            When called on a dataframe with no `col` specified or when called
            on an object other than a dataframe or series.

        """
        match df:
            case DataFrame():
                return df.explode(self.cols, ignore_index=self.ignore_index)
            case Series():
                return df.explode(ignore_index=self.ignore_index)
            case _:
                cls = type(df).__name__
                tmp = 'Cannot explode an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

    @staticmethod
    def __valid(cols: Labels) -> list[Hashable]:
        """Ensure that the columns are indeed a sequence of hashables."""
        if isinstance(cols, str):
            return [cols]
        try:
            _ = [hash(col) for col in cols]
        except TypeError:
            _ = hash(cols)
            return [cols]
        return list(cols)

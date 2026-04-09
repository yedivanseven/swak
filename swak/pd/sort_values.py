from typing import Any, overload
from collections.abc import Hashable
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Labels


class SortValues(ArgRepr):
    """Partial of the pandas dataframe or series ``sort_values`` method.

    Parameters
    ----------
    by: hashable or sequence
        Name or list of names to sort by. Ignored if used with a series.
    *bys: hashable
        Additional names to sort by. Again ignored if used with a series.
    **kwargs
        Additional keyword arguments will be forwarded to the method call with
        the exception of "inplace", which will be set to ``False``.

    Note
    ----
    For a full list of keyword arguments and their description, see the
    pandas `sort_values documentation <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.DataFrame.sort_values.html>`_.

    """

    def __init__(
            self,
            by: Labels,
            *bys: Hashable,
            **kwargs: Any
    ) -> None:
        self.bys = self.__valid(by) + self.__valid(bys)
        self.kwargs = (kwargs.pop('inplace', ''), kwargs)[1]
        super().__init__(self.bys, **self.kwargs)


    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Sort a pandas dataframe or series by column(s) values.

        Parameters
        ----------
        df: DataFrame or Series
            The dataframe or series to sort.

        Returns
        -------
        DataFrame or Series
            The sorted dataframe or series.

        Raises
        ------
        TypeError
            If called on anything else other than a pandas series or dataframe.

        """
        match df:
            case DataFrame():
                return df.sort_values(self.bys, inplace=False, **self.kwargs)
            case Series():
                return df.sort_values(inplace=False, **self.kwargs)
            case _:
                cls = type(df).__name__
                tmp = '"df" must be a pandas DataFrame or Series, not a {}!'
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

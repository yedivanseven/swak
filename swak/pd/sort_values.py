from typing import Any
from functools import singledispatchmethod
from collections.abc import Hashable, Sequence
from pandas import DataFrame, Series
from ..misc import ArgRepr


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
            by: Hashable | Sequence[Hashable],
            *bys: Hashable,
            **kwargs: Any
    ) -> None:
        self.bys = self.__valid(by) + self.__valid(bys)
        self.kwargs = (kwargs.pop('inplace', ''), kwargs)[1]
        super().__init__(*self.bys, **kwargs)


    @singledispatchmethod
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
        cls = type(df).__name__
        tmp = '"df" must be a pandas DataFrame or Series, not a {}!'
        msg = tmp.format(cls)
        raise TypeError(msg)

    @__call__.register
    def _(self, df: DataFrame) -> DataFrame:
        return  df.sort_values(self.bys, inplace=False, **self.kwargs)

    @__call__.register
    def _(self, df: Series) -> Series:
        return  df.sort_values(inplace=False, **self.kwargs)

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

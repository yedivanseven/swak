from typing import Any
from pandas import DataFrame, Series
from ..misc import ArgRepr

type Other = Series | DataFrame


class Join(ArgRepr):
    """Light wrapper around the pandas dataframe ``join`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``join`` method call.
    **kwargs
        Keyword arguments to pass on to the ``join`` method call.

    Note
    ----
    For a full list of (keyword) arguments and their description, see the
    pandas `join documentation <https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.join.html>`_.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df: DataFrame, *others: Other) -> DataFrame:
        """Join a dataframe with other dataframe(s) and/or series.

        Parameters
        ----------
        df: DataFrame
            Source dataframe on which the ``join`` method will be called.
        *others: DataFrame or Series
            Index should be similar to one (or more) columns in `df`. If a
            series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined dataframe.

        Returns
        -------
        DataFrame
            The joined dataframe.

        """
        if len(others) == 0:
            return df
        elif len(others) == 1:
            return df.join(others[0], *self.args, **self.kwargs)
        else:
            return df.join(others, *self.args, **self.kwargs)

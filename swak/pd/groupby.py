from typing import Any
from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from ..misc import ArgRepr


class FrameGroupByAgg(ArgRepr):
    """Simple partial of for calling a grouped dataframe's ``agg`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``groupby`` method call.
    **kwargs
        Keyword arguments to pass on to the ``groupby`` method call.

    Notes
    -----
    See the pandas `agg docs <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html>`_ for
    a full list of (keyword) arguments and an extensive description of
    usage and configuration.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, grouped_df: DataFrameGroupBy) -> DataFrame:
        """Call a grouped dataframe`s ``agg`` method with the cached (kw)args.

        Parameters
        ----------
        grouped_df: DataFrameGroupBy
            The grouped dataframe to aggregate.

        Returns
        -------
        DataFrame
            The aggregation of the grouped dataframe.

        """
        return grouped_df.agg(*self.args, **self.kwargs)

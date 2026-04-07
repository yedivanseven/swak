from typing import Any, overload
from pandas.core.window import Rolling, Window, RollingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series
from ..misc import ArgRepr


class RollingWindow(ArgRepr):
    """Simple partial of for calling a pandas object's ``rolling`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``rolling`` method call.
    **kwargs
        Keyword arguments to pass on to the ``rolling`` method call.

    Notes
    -----
    See the pandas `rolling docs <https://pandas.pydata.org/docs/dev/reference
    /api/pandas.core.groupby.DataFrameGroupBy.rolling.html>`_ for a full list
    of (keyword) arguments and an extensive description of usage.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    @overload
    def __call__(self, df: Series) -> Rolling | Window:
        ...

    @overload
    def __call__(self, df: DataFrame) -> Rolling | Window:
        ...

    @overload
    def __call__(self, df: SeriesGroupBy) -> RollingGroupby:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> RollingGroupby:
        ...

    def __call__(self, df):
        """Call a pandas object`s ``rolling`` method with the cached (kw)args.

        Parameters
        ----------
        df: Series, DataFrame, or their GroupBy companions
            The pandas object to call ``rolling`` on.

        Returns
        -------
        Window, Rolling, or RollingGroupBy
            Depending on the input type.

        """
        return df.rolling(*self.args, **self.kwargs)

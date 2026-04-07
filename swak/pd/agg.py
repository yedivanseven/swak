from typing import Any, overload
from collections.abc import Callable
from pandas.core.window import Rolling, RollingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series
from ..misc import ArgRepr

type Func = Callable[[Series], float] | str
type Funcs = list[Func] | dict[str, Func] | dict[str, list[Func]]
list

class Agg(ArgRepr):
    """Simple partial for calling a pandas object's ``agg`` method.

    Parameters
    ----------
    func: callable, str, list, or dict, optional
        Function(s) to use for aggregating the data. If a function, must work
        when passed a Series. Also acceptable are a function name, a list of
        function names and a dictionary with columns names as keys and
        functions, function names, or lists thereof as values.
        Defaults to ``None``, which only works for a dataframe and relies on
        `kwargs` to specify `named aggregations <https://pandas.pydata.org/
        pandas-docs/stable/user_guide/groupby.html#groupby-aggregate-named>`_.
    *args
        Positional arguments to pass on to the ``agg`` or `func` call.
    **kwargs
        Keyword arguments to pass on to the ``agg`` or `func` call.

    Note
    ----
    See the pandas `agg docs <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html>`_ for
    a full list of (keyword) arguments and an extensive description of
    usage and configuration.

    """

    def __init__(
            self,
            func: Func | Funcs | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(func, *args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @overload
    def __call__(self, df: DataFrame) -> Series | DataFrame:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> DataFrame:
        ...

    @overload
    def __call__(self, df: Series) -> Any:
        ...

    @overload
    def __call__(self, df: SeriesGroupBy) -> Series:
        ...

    @overload
    def __call__(self, df: Rolling) -> Series | DataFrame:
        ...

    @overload
    def __call__(self, df: RollingGroupby) -> Series | DataFrame:
        ...

    def __call__(self, df):
        """Call a pandas object's ``agg`` method with the cached (kw)args.

        Parameters
        ----------
        df: Series, DataFrame, Rolling or their GroupBy companions
            The pandas object to aggregate.

        Returns
        -------
        scalar, Series, or DataFrame
            The aggregation of the pandas object.

        """
        return df.agg(self.func, *self.args, **self.kwargs)

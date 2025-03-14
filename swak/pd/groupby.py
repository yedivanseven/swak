from typing import Any, Literal
from collections.abc import Callable
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import RollingGroupby
from ..misc import ArgRepr

type Func = Callable[[Series], float] | str
type Funcs = list[Func] | dict[str, Func] | dict[str, list[Func]]


class FrameGroupByAgg(ArgRepr):
    """Simple partial for calling a grouped dataframe's ``agg`` method.

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


class RollingGroupBy(ArgRepr):
    """Simple partial of for calling a grouped dataframe's ``rolling`` method.

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

    def __call__(self, grouped_df: DataFrameGroupBy) -> RollingGroupby:
        """Call a grouped dataframe`s ``rolling`` method with cached (kw)args.

        Parameters
        ----------
        grouped_df: DataFrameGroupBy
            The grouped dataframe to call ``rolling`` on.

        Returns
        -------
        RollingGroupby
            The rolling-grouped dataframe.

        """
        return grouped_df.rolling(*self.args, **self.kwargs)


class RollingGroupByAgg(ArgRepr):
    """Simple partial for calling a rolling-grouped dataframe's ``agg`` method.

    Parameters
    ----------
    func: callable, str, list, or dict
        Function(s) to use for aggregating the data. If a function, must work
        when passed a Series. Also acceptable are a function name, a list of
        function names and a dictionary with columns names as keys and
        functions, function names, or lists thereof as values.
    *args
        Positional arguments to pass to `func`.
    **kwargs
         Keyword arguments to pass to `func`.

    """

    def __init__(self, func: Funcs, *args: Any, **kwargs: Any) -> None:
        super().__init__(func, *args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, rolling_df: RollingGroupby) -> DataFrame:
        """Call a rolling-grouped dataframe`s ``agg`` method.

        Parameters
        ----------
        rolling_df: RollingGroupby
            The rolling-grouped dataframe to aggregate.

        Returns
        -------
        DataFrame
            The aggregation of the rolling-grouped dataframe.

        """
        return rolling_df.agg(self.func, *self.args, **self.kwargs)


class RollingGroupByApply(ArgRepr):
    """Partial for calling a rolling-grouped dataframe's ``apply`` method.

    Parameters
    ----------
    func: callable
        Must produce a single numerical value from a numpy ndarray if
        `raw` = ``True`` or a series if `raw` = ``False``.
        Can also accept a numba JIT function with `engine` = "numba" specified.
    raw: bool, optional
        Whether to pass a numpy ndarray or a pandas series to `func`.
        Defaults to ``False``.
    engine: str, optional
        Either "cython" or "numba". Defaults to ``None``
    engine_kws: dict, optional
        Configuration of the "numba" engine. Keys can be "nopython", "nogil",
        and "parallel", and values must be ``True`` or ``False``.
        Defaults to ``None``.
    *args
        Positional arguments to pass to `func`.
    **kwargs
         Keyword arguments to pass to `func`.

    """

    def __init__(
            self,
            func: Callable[[Series | ndarray], float],
            raw: bool = False,
            engine: Literal['cython', 'numba'] | None = None,
            engine_kws: dict[str, bool] | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(func, raw, engine, engine_kws, *args, **kwargs)
        self.func = func
        self.raw = raw
        self.engine = engine
        self.engine_kws = engine_kws
        self.args = args
        self.kwargs = kwargs

    def __call__(self, rolling_df: RollingGroupby) -> DataFrame:
        """Call a rolling-grouped dataframe`s ``apply`` method.

        Parameters
        ----------
        rolling_df: RollingGroupby
            The rolling-grouped dataframe to aggregate.

        Returns
        -------
        DataFrame
            The aggregation of the rolling-grouped dataframe.

        """
        return rolling_df.apply(
            self.func,
            self.raw,
            self.engine,
            self.engine_kws,
            self.args,
            self.kwargs
        )

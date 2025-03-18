from typing import Any, Literal
from collections.abc import Callable
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.window import RollingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from ..misc import ArgRepr

type GroupBy = SeriesGroupBy | DataFrameGroupBy
type Pandas = Series | DataFrame


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


class GroupByApply(ArgRepr):
    """Partial for calling a grouped dataframe or series ``apply`` method.

    Parameters
    ----------
    func: callable
        A callable that takes a dataframe or series as its first argument,
        and returns a dataframe, a series or a scalar.
    *args
        Positional arguments to pass to `func`.
    **kwargs
         Keyword arguments to pass to `func`.

    """

    def __init__(
            self,
            func: Callable[[Pandas, ...], Any],
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(func, *args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df: GroupBy) -> Pandas:
        """Call a grouped dataframe or series ``apply`` method.

        Parameters
        ----------
        df: DataFrameGroupBy or SeriesGroupBy
            The pandas group-by object to apply `func` to.

        Returns
        -------
        Series or DataFrame
            The input object with `func` applied to groups.

        """
        return df.apply(self.func, *self.args, **self.kwargs)

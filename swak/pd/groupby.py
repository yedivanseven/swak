from typing import Any, Literal, overload
from collections.abc import Callable, Hashable, Sequence, Mapping
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.window import RollingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy, Grouper
from ..misc import ArgRepr

type GroupByT = SeriesGroupBy | DataFrameGroupBy
type Pandas = Series | DataFrame
type GroupKey = (
    str
    | Callable[[Hashable], Hashable]
    | Grouper
    | Mapping[Hashable, Hashable]
    | ndarray[tuple[int], np.dtype[Any]]
    | Series
)
type GroupKeys = list[GroupKey]


class GroupBy(ArgRepr):
    """Simple partial of a pandas dataframe or series ``groupby`` method.

    Parameters
    ----------
    by: str, callable, series, array, dict, or list
        Column name, function (to be called on each column name), list or numpy
        array of the same length as the columns, a dict or series providing a
        label -> group name mapping, or a list of the above.
    level: hashable or sequence, optional
        If the axis is a multi-index (hierarchical), group by a particular
        level or levels. Do not specify both `by` and `level`.
        Defaults to ``None``.
    as_index: bool, optional
        Whether to return group labels as index. Defaults to ``True``.
    sort: bool, optional
        Whether to sort group keys. Defaults to ``True``.
    group_keys: bool, optional
        Defaults to ``True``
    observed: bool, optional
        Whether to show only observed values for categorical groupers.
        Defaults to ``False``.
    dropna: bool, optional
        Whether to treat NA values in group keys as groups.
        Defaults to ``True``.

    Note
    ----
    For a more extensive description of all (keyword) arguments, see the
    pandas `documentation <https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.groupby.html>`_.

    """

    def __init__(
            self,
            by: GroupKey | GroupKeys | None = None,
            level: Hashable | Sequence[Hashable] | None = None,
            as_index: bool = True,
            sort: bool = True,
            group_keys: bool = True,
            observed: bool = False,
            dropna: bool = True,
    ) -> None:
        super().__init__(
            by,
            level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna
        )
        self.by = by
        self.level = level
        self.as_index = as_index
        self.sort = sort
        self.group_keys = group_keys
        self.observed = observed
        self.dropna = dropna

    @overload
    def __call__(self, df: Series) -> SeriesGroupBy:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrameGroupBy:
        ...

    def __call__(self, df):
        """Call a dataframe or series ``groupby`` method.

        Parameters
        ----------
        df: DataFrame or Series
            Pandas dataframe or series to group.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            The grouped dataframe or series.

        """
        return df.groupby(
            self.by,
            self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            observed=self.observed,
            dropna=self.dropna
        )


# ToDo: Make these work for Series and DataFrame as well and move out of here!
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

    def __call__(self, df: GroupByT) -> Pandas:
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

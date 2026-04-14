from typing import Any, Literal, overload
from collections.abc import Mapping
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
from pandas import DataFrame, Series
from pandas._typing import AggFuncType
from ..misc import ArgRepr
from .types import Axis


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
    axis: int or str, optional
        Which dimension to aggregate over in case of a dataframe. Must be one
        of 0, "index", 1, or "columns". Ignored for all other pandas objects.
        Defaults to 0.
    *args
        Positional arguments to pass on to the ``agg`` or `func` call.
    engine: str, optional
        Which engine to use when applied to group-by objects.
    engine_kwargs: dict, optional
        Keywords to configure the engine, if any.
    **kwargs
        Keyword arguments to pass on to the ``agg`` or `func` call. When used
        on a dataframe or series, and if `func` is ``None``, column names with
        their individual aggregation functions can be given.

    Note
    ----
    See the pandas `agg docs <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html>`_ for
    a full list of (keyword) arguments and an extensive description of
    usage and configuration.

    """

    def __init__(
            self,
            func: AggFuncType | None = None,
            axis: Axis = 0,
            *args: Any,
            engine: Literal['cython', 'numba'] | None = None,
            engine_kwargs: Mapping[str, bool] | None = None,
            **kwargs: Any
    ) -> None:
        self.func = func
        self.axis = axis
        self.args = args
        self.engine = engine
        self.engine_kwargs = engine_kwargs
        self.kwargs = kwargs
        name = func if callable(func) or func is None else type(func)
        super().__init__(
            name,
            axis,
            *args,
            engine=engine,
            engine_kwargs=engine_kwargs,
            **kwargs
        )

    @overload
    def __call__(self, df: DataFrame) -> Series | DataFrame:
        ...

    @overload
    def __call__(self, df: Series) -> Series | Any:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> DataFrame:
        ...

    @overload
    def __call__(self, df: SeriesGroupBy) -> Series:
        ...

    def __call__(self, df: Any) -> DataFrame | Series | Any:
        """Call a pandas object's ``agg`` method with the cached (kw)args.

        Parameters
        ----------
        df
            The pandas object to aggregate.

        Returns
        -------
        Scalar, Series, or DataFrame
            The aggregation of the pandas object.

        """
        match df:
            case DataFrame():
                return df.agg(
                    self.func,
                    self.axis,
                    *self.args,
                    **self.kwargs
                )
            case Series():
                return df.agg(
                    self.func,
                    0,
                    *self.args,
                    **self.kwargs
                )
            case DataFrameGroupBy() | SeriesGroupBy():
                return df.agg(
                    self.func,
                    *self.args,
                    engine=self.engine,
                    engine_kwargs=self.engine_kwargs,
                    **self.kwargs
                )
            case _:
                return df.agg(self.func,*self.args, **self.kwargs)

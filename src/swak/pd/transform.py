from typing import Any, Literal, overload
from collections.abc import Mapping
from pandas.core.groupby import SeriesGroupBy, DataFrameGroupBy
from pandas.core.resample import Resampler
from pandas import DataFrame, Series
from pandas._typing import AggFuncType
from ..misc import ArgRepr
from .types import Axis


class Transform(ArgRepr):
    """Simple partial for calling a pandas object's ``transform`` method.

    Parameters
    ----------
    func: callable, str, list, or dict, optional
        Function(s) to use for transforming the data.
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
    See the pandas `transform docs <https://pandas.pydata.org/docs/reference
    /api/pandas.api.typing.DataFrameGroupBy.transform.html>`_ for a full list
    of (keyword) arguments and  description of usage and configuration.

    """

    def __init__(
            self,
            func: AggFuncType,
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
        name = func if callable(func) else type(func)
        super().__init__(
            name,
            axis,
            *args,
            engine=engine,
            engine_kwargs=engine_kwargs,
            **kwargs
        )

    @overload
    def __call__(self, df: DataFrame | DataFrameGroupBy) -> DataFrame:
        ...

    @overload
    def __call__(self, df: Series | SeriesGroupBy) -> Series:
        ...

    @overload
    def __call__(self, df: Resampler) -> Series:
        ...

    def __call__(self, df):
        """Call a pandas object's ``transform`` method with cached (kw)args.

        Parameters
        ----------
        df: Series, DataFrame, their GroupBy companions, or Resampler
            The pandas object to transform.

        Returns
        -------
        Series or DataFrame
            Depending on the input type.

        Raises
        ------
        TypeError
            When called on an unsuitable object type.

        """
        match df:
            case DataFrame():
                return df.transform(
                    self.func,
                    self.axis,
                    *self.args,
                    **self.kwargs
                )
            case Series():
                return df.transform(
                    self.func,
                    0,
                    *self.args,
                    **self.kwargs
                )
            case DataFrameGroupBy() | SeriesGroupBy():
                return df.transform(
                    self.func,
                    *self.args,
                    engine=self.engine,
                    engine_kwargs=self.engine_kwargs,
                    **self.kwargs
                )
            case Resampler():
                return df.transform(self.func, *self.args, **self.kwargs)
            case _:
                cls = type(df).__name__
                tmp = 'Cannot transform an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

from typing import Any, Literal
from functools import singledispatchmethod
from pandas import DataFrame, Series
from pandas.core.window.rolling import BaseWindow
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.groupby import DataFrameGroupBy
from pandas._typing import AggFuncType
from ..misc import ArgRepr
from .types import Axis

type Result = Literal['expand', 'reduce', 'broadcast']
type Pandas = (
    DataFrame
    | Series
    | DataFrameGroupBy
    | BaseGroupBy
    | BaseWindow
)


class Apply(ArgRepr):
    """Simple partial for calling a pandas object's ``apply`` method.

    Parameters
    ----------
    func: callable, str, list, or dict
        Function(s) to apply to the data.
    axis: int or str, optional
        Which dimension to apply `func` over in case of a dataframe. Must be
        one of 0, "index", 1, or "columns". Ignored for all other pandas
        objects. Defaults to 0.
    raw: bool, optional
        Whether to pass a series or a numpy array to `func`.
        Defaults to ``False``, which results in a series being passed.
    result_type: str, optional
        Must be one of "expand", "reduce", "broadcast", or ``None``.
    args: tuple, optional
        Positional arguments to pass on to `func`. Defaults to an emtpy tuple.
    by_row: str or bool, optional
        Must be one of "compat" or ``False``.
    engine: str or decorator, optional
        Which engine to use. Defaults to the python interpreter.
    engine_kwargs: dict, optional
        Keywords to configure the engine, if any.
    **kwargs
        Keyword arguments to pass on to the `func` call.

    Note
    ----
    See the pandas `apply docs <https://pandas.pydata.org/docs/reference/api/
    pandas.DataFrame.apply.html#pandas.DataFrame.apply>`_ for a full list of
    (keyword) arguments and a description of usage and configuration.

    """

    def __init__(
            self,
            func: AggFuncType,
            axis: Axis = 0,
            raw: bool = False,
            result_type: Result | None = None,
            args: tuple[Any, ...] = (),
            by_row: Literal['compat', False] = 'compat',
            engine: Literal['cython', 'numba', 'python'] | None = None,
            engine_kwargs: dict[str, bool] | None = None,
            **kwargs: Any
    ) -> None:
        self.func = func
        self.axis = axis
        self.raw = raw
        self.result_type = result_type
        self.args = args
        self.by_row = by_row
        self.engine = engine
        self.engine_kwargs = engine_kwargs
        self.kwargs = kwargs
        name = func if callable(func) else type(func)
        super().__init__(
            name,
            axis=self.axis,
            raw=self.raw,
            result_type=self.result_type,
            args=self.args,
            by_row=self.by_row,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )

    @singledispatchmethod
    def __call__(self, df) -> Any:
        """Call a pandas object's ``apply`` method.

        Parameters
        ----------
        df: Pandas
            The pandas object to call ``apply`` on.

        Returns
        -------
        Pandas
            The return type of calling ``apply`` on the pandas object.

        """
        return df.apply(self.func, *self.args, **self.kwargs)

    @__call__.register
    def _(self, df: DataFrame) -> Series | DataFrame:
        return df.apply(
            self.func,
            axis=self.axis,
            raw=self.raw,
            result_type=self.result_type,
            args=self.args,
            by_row=self.by_row,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )

    @__call__.register
    def _(self, df: Series) -> Series | DataFrame:
        return df.apply(
            self.func,
            args=self.args,
            by_row=self.by_row,
            **self.kwargs
        )

    @__call__.register
    def _(self, df: DataFrameGroupBy) -> Series | DataFrame:
        return df.apply(
            self.func,
            *self.args,
            include_groups=False,
            **self.kwargs
        )

    @__call__.register
    def _(self, df: BaseWindow) -> Series | DataFrame:
        return df.apply(
            self.func,
            raw=self.raw,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            args=self.args,
            kwargs=self.kwargs
        )

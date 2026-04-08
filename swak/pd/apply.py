from typing import Any, Literal
from collections.abc import Callable, Mapping
from functools import singledispatchmethod
from pandas import DataFrame, Series
from pandas.core.window import RollingGroupby, ExpandingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from ..misc import ArgRepr
from .types import Axis, Engine

type Result = Literal['expand', 'reduce', 'broadcast']
type Pandas = (
    DataFrame
    | Series
    | DataFrameGroupBy
    | SeriesGroupBy
    | RollingGroupby
    | ExpandingGroupby
)

class Apply(ArgRepr):
    # ToDo. Write docstring

    def __init__(
            self,
            func: Callable,  # ToDo: This can also be str, list, or dict
            axis: Axis = 0,
            raw: bool = False,
            result_type: Result | None = None,
            args: tuple[Any, ...] = (),
            by_row: Literal['compat', False] = 'compat',
            engine: Engine = 'python',
            engine_kwargs: Mapping[str, bool] | None = None,
            include_groups: bool = True,
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
        self.include_groups = include_groups
        self.kwargs = kwargs
        super().__init__(
            self.func,
            axis=self.axis,
            raw=self.raw,
            result_type=self.result_type,
            args=self.args,
            by_row=self.by_row,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            include_groups=self.include_groups,
            **self.kwargs
        )

    @singledispatchmethod
    def __call__(self, df):
        """Call a pandas object's ``apply`` method.

        Parameters
        ----------
        df: Pandas
            The pandas object to call ``apply`` on.

        Returns
        -------
        Pandas
            The return type of calling ``apply`` on teh pandas object.

        Raises
        ------
        TypeError
            When called with an unsuitable object type.

        """
        cls = type(df).__name__
        tmp = 'Cannot call "apply" on an object of type {}!'
        msg = tmp.format(cls)
        raise TypeError(msg)

    # ToDo: Can we not make these show up in the sphinx doc?
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
            include_groups=self.include_groups,
            **self.kwargs
        )

    @__call__.register
    def _(self, df: SeriesGroupBy) -> Series | DataFrame:
        return df.apply(
            self.func,
            *self.args,
            **self.kwargs
        )

    @__call__.register
    def _(self, df: RollingGroupby) -> Series | DataFrame:
        return df.apply(
            self.func,
            raw=self.raw,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            args=self.args,
            kwargs=self.kwargs
        )

    @__call__.register
    def _(self, df: ExpandingGroupby) -> Series | DataFrame:
        return df.apply(
            self.func,
            raw=self.raw,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            args=self.args,
            kwargs=self.kwargs
        )

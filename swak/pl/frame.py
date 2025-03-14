from typing import Any
from collections.abc import Iterable, Sequence
from datetime import timedelta
from numpy import ndarray
from pandas import DataFrame as PandasFrame
from polars.dataframe.group_by import GroupBy as GroupByT, DynamicGroupBy
from polars import DataFrame, Series, Expr
from polars._typing import (
    IntoExprColumn,
    IntoExpr,
    ColumnNameOrSelector,
    StartBy,
    ClosedInterval,
    Label,
    JoinStrategy,
    JoinValidation,
    MaintainOrderJoin
)
from ..misc import ArgRepr
from .types import IntoExprs

type Column = ColumnNameOrSelector | Iterable[ColumnNameOrSelector]
type Predicate = (
    bool
    | list[bool]
    | ndarray[bool]
    | Series
    | IntoExprColumn
    | Iterable[IntoExprColumn]
)


class Select(ArgRepr):
    """Partial of the polars dataframe `select <select_>`__ method.

    Parameters
    ----------
    *exprs: IntoExpr
        Column(s) to select, specified as positional arguments. Accepts
        expression input. Strings are parsed as column names, other
        non-expression inputs are parsed as literals.
    **named_exprs: IntoExpr
        Additional columns to select, specified as keyword arguments. The
        columns will be renamed to the keyword used.


    .. _select: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.select.html

    """

    def __init__(self, *exprs: IntoExprs, **named_exprs: IntoExpr) -> None:
        super().__init__(*exprs, **named_exprs)
        self.exprs = exprs
        self.named_exprs = named_exprs

    def __call__(self, df: DataFrame) -> DataFrame:
        """Select columns from a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The Dataframe to select columns from.

        Returns
        -------
        DataFrame
            The selected columns.

        """
        return df.select(*self.exprs, **self.named_exprs)


class Filter(ArgRepr):
    """Partial of the polars dataframe `filter <filter_>`__ method.

    Parameters
    ----------
    *predicates:
        Expression(s) that evaluate to a boolean Series.
    **constraints
        Filter column(s) given named by the keyword argument itself by the
        supplied value. Constraints will be implicitly combined with other
        filters with a logical and.


    .. _filter: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.filter.html

    """

    def __init__(self, *predicates: Predicate, **constraints: Any) -> None:
        super().__init__(*predicates, **constraints)
        self.predicates = predicates
        self.constraints = constraints

    def __call__(self, df: DataFrame) -> DataFrame:
        """Filter dataframe by `predicates` anf value `constraints`.

        Parameters
        ----------
        df: DataFrame
            The dataframe to filter.

        Returns
        -------
        DataFrame
            The filtered dataframe.

        """
        return df.filter(*self.predicates, **self.constraints)


class Drop(ArgRepr):
    """Partial of the polars dataframe `drop <drop_>`__ method.

    Parameters
    ----------
    *columns: ColumnNameOrSelector
        Names of the columns that should be removed from the dataframe.
        Accepts column selector input.
    strict: bool, optional
        Validate that all column names exist in the current schema, and throw
        an exception if any do not. Defaults to ``True``


    .. _drop: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.drop.html

    """

    def __init__(self, *columns: Column, strict: bool = True) -> None:
        super().__init__(*columns, strict=strict)
        self.columns = columns
        self.strict = strict

    def __call__(self, df: DataFrame) -> DataFrame:
        """Drop `columns` from a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The Dataframe to drop columns from.

        Returns
        -------
        DataFrame
            The Dataframe without the dropped `columns`.

        """
        return df.drop(*self.columns, strict=self.strict)


class Sort(ArgRepr):
    """Partial of the polars dataframe `sort <sort_>`__ method.

    Parameters
    ----------
    by: IntoExpr
        Column(s) to sort by. Accepts expression input, including selectors.
        Strings are parsed as column names.
    *more_by: IntoExpr
        Additional columns to sort by, specified as positional arguments.
    descending: bool, optional
        Sort in descending order. When sorting by multiple columns, can be
        specified per column by passing a sequence of booleans.
        Defaults to ``False``.
    nulls_last: bool, optional
        Place null values last. Can be a single boolean applying to all
        columns or a sequence of booleans for per-column control.
        Defaults to ``False``
    multithreaded: bool, optional
        Sort using multiple threads. Defaults to ``True``.
    maintain_order: bool, optional
        Whether the order should be maintained if elements are equal.
        Defaults to ``False``.


    .. _sort: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.sort.html

    """

    def __init__(
            self,
            by: IntoExpr | Iterable[IntoExpr],
            *more_by: IntoExpr,
            descending: bool | Sequence[bool] = False,
            nulls_last: bool | Sequence[bool] = False,
            multithreaded: bool = True,
            maintain_order: bool = False,
    ) -> None:
        super().__init__(
            by,
            *more_by,
            descending=descending,
            nulls_last=nulls_last,
            multithreaded=multithreaded,
            maintain_order=maintain_order
        )
        self.by = by
        self.more_by = more_by
        self.descending = descending
        self.nulls_last = nulls_last
        self.multithreaded = multithreaded
        self.maintain_order = maintain_order

    def __call__(self, df: DataFrame) -> DataFrame:
        """Sort polars dataframe by column values.

        Parameters
        ----------
        df: DataFrame
            The Dataframe to sort.

        Returns
        -------
        DataFrame
            The sorted dataframe.

        """
        return df.sort(
            self.by,
            *self.more_by,
            descending=self.descending,
            nulls_last=self.nulls_last,
            multithreaded=self.multithreaded,
            maintain_order=self.maintain_order
        )


class WithColumns(ArgRepr):
    """Partial of the polars dataframe `with_columns <with_>`__ method.

    Parameters
    ----------
    *exprs: IntoExpr
        Column(s) to add, specified as positional arguments. Accepts expression
        input. Strings are parsed as column names, other non-expression inputs
        are parsed as literals.
    **named_exprs: IntoExpr
        Additional columns to add, specified as keyword arguments. The columns
        will be renamed to the keyword used.


    .. _with: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.with_columns.html

    """

    def __init__(self, *exprs: IntoExprs, **named_exprs: IntoExpr) -> None:
        super().__init__(*exprs, **named_exprs)
        self.exprs = exprs
        self.named_exprs = named_exprs

    def __call__(self, df: DataFrame) -> DataFrame:
        """Add or replace columns to/of a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to add or replace columns to/of.

        Returns
        -------
        DataFrame
            The dataframe with columns added or replaced.

        """
        return df.with_columns(*self.exprs, **self.named_exprs)


class GroupBy(ArgRepr):
    """Partial of the polars dataframe `group_by <group_>`__ method.

    Parameters
    ----------
    *by: IntoExpr
        Column(s) to group by. Accepts expression input. Strings are parsed as
        column names.
    maintain_order: bool, optional
        Ensure that the order of the groups is consistent with the input data.
        This is slower than a default group by. Settings this to True blocks
        the possibility to run on the streaming engine. Default to ``False``.
    **named_by: IntoExpr
        Additional columns to group by, specified as keyword arguments.
        The columns will be renamed to the keyword used.


    .. _group: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
               polars.DataFrame.group_by.html

    """

    def __init__(
            self,
            *by: IntoExprs,
            maintain_order: bool = False,
            **named_by: IntoExpr,
    ) -> None:
        super().__init__(*by, maintain_order=maintain_order, **named_by)
        self.by = by
        self.maintain_order = maintain_order
        self.named_by = named_by

    def __call__(self, df: DataFrame) -> GroupByT:
        """Group a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to group.

        Returns
        -------
        DataFrame
            The grouped dataframe.

        """
        return df.group_by(
            *self.by,
            maintain_order=self.maintain_order,
            **self.named_by
        )


class GroupByDynamic(ArgRepr):
    """Partial of the polars dataframe `group_by_dynamic <dyngrp_>`__ method.

    Parameters
    ----------
    index_column: IntoExpr
        Column used to group based on the time window. Often of type Date or
        Datetime. This column must be sorted in ascending order (or, if
        group_by is specified, then it must be sorted in ascending order
        within each group). In case of a dynamic group by on indices, dtype
        needs to be Int32 or Int64. Note that Int32 gets temporarily cast to
        Int64, so if performance matters use an Int64 column.
    every: str or timedelta
        Interval of the window. Suffix string of integer number with the
        letter "i" to indicate indexing by integer columns.
    period: str or timedelta, optional
        Length of the window. Equals ‘every’ if set to ``None`` (the default).
    offset: str or timedelta
        Offset of the window. Does not take effect if `start_by` is
        "datapoint". Defaults to zero.
    include_boundaries: bool, optional
        Add the lower and upper bound of the window to the "_lower_boundary"
        and “_upper_boundary” columns. This will impact performance because it
        is harder to parallelize. Defaults to ``False``.
    closed: "left", "right", "both", "none"
        Define which sides of the temporal interval are closed (inclusive).
    label: "left", "right", "datapoint"
        Which label to use for the window, lower boundary, upper boundary, or
        first value of the index column in the given window. If you don't need
        the label to be at one of the boundaries, choose this option for
        maximum performance.
    group_by: IntoExpr, optional
        Also group by this column/these columns. Defaults to ``None``.
    start_by: "window", "datapoint", "monday", "tuesday", ...
        The strategy to determine the start of the first window by, where
        "window" takes the earliest timestamp, truncates it with `every`, and
        then adds `offset`. Weekly windows start on Monday. "datapoint" starts
        from the first encountered data point, whereas any day of the week
        starts the window at the weekday before the first data point.
        The resulting window is then shifted back until the earliest datapoint
        is in or in front of it.


    .. _dyngrp: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.group_by_dynamic.html

    """

    def __init__(
            self,
            index_column: IntoExpr,
            every: str | timedelta,
            period: str | timedelta | None = None,
            offset: str | timedelta | None = None,
            include_boundaries: bool = False,
            closed: ClosedInterval = 'left',
            label: Label = 'left',
            group_by: IntoExprs | None = None,
            start_by: StartBy = 'window',
    ) -> None:
        self.index_column = index_column
        self.every = every
        self.period = period
        self.offset = offset
        self.include_boundaries = include_boundaries
        self.closed = closed.strip().lower()
        self.label = label.strip().lower()
        self.group_by = group_by
        self.start_by = start_by.strip().lower()
        super().__init__(
            index_column,
            every=every,
            period=period,
            offset=offset,
            closed=self.closed,
            label=self.label,
            group_by=group_by,
            start_by=self.start_by,
        )

    def __call__(self, df: DataFrame) -> DynamicGroupBy:
        """Evaluate rolling-window aggregations on a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to compute rolling-window aggregations on.

        Returns
        -------
        DataFrame
            The rolling-window aggregations.

        """
        return df.group_by_dynamic(
            self.index_column,
            every=self.every,
            period=self.period,
            offset=self.offset,
            closed=self.closed,
            label=self.label,
            group_by=self.group_by,
            start_by=self.start_by
        )


class Join(ArgRepr):
    """Partial of the polars dataframe `join <join_>`__ method.

    Parameters
    ----------
    on: str
        Name(s) of the join columns in both DataFrames. If set, `left_on` and
        `right_on` should be ``None``. Should not be specified if `how` is
        "cross". Defaults to ``None``.
    how: "inner", "left", "right", "full", "semi", "anti", "cross"
        Join strategy.
    left_on: str, optional
        Name(s) of the left join column(s). Defaults to ``None``.
    right_on: str, optional
        Name(s) of the right join column(s). Defaults to ``None``.
    suffix: str, optional
        Suffix to append to columns with a duplicate name.
        Defaults to "_right".
    validate: "m:m", "m:1", "1:m", "1:1"
        Checks if join is of specified type, many-to-many, many-to-one,
        one-to_many, or one-to-one.
    nulls_equal: bool, optional
        Join on null values. By default, null values will never produce
        matches. Defaults to ``False``.
    coalesce: bool, optional
        Coalescing behavior (merging of join columns). Defaults to ``None``,
        which leaves the behaviour join specific.
    maintain_order: "none", "left", "right", "left_right", "right_left"
        Which dataframe row order to preserve, if any. Do not rely on any
        observed ordering without explicitly setting this parameter, as your
        code may break in a future release. Not specifying any ordering can
        improve performance Supported for inner, left, right and full joins.


    .. _join: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.join.html

    """

    def __init__(
            self,
            on: str | Expr | Sequence[str | Expr] | None = None,
            how: JoinStrategy = 'inner',
            left_on: str | Expr | Sequence[str | Expr] | None = None,
            right_on: str | Expr | Sequence[str | Expr] | None = None,
            suffix: str = '_right',
            validate: JoinValidation = 'm:m',
            nulls_equal: bool = False,
            coalesce: bool | None = None,
            maintain_order: MaintainOrderJoin | None = None,
    ) -> None:
        self.on = on
        self.how = how.strip().lower()
        self.left_on = left_on
        self.right_on = right_on
        self.suffix = suffix.strip()
        self.validate = validate.strip().lower()
        self.nulls_equal = nulls_equal
        self.coalesce = coalesce
        self.maintain_order = maintain_order
        super().__init__(
            on,
            self.how,
            left_on=left_on,
            right_on=right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
        )

    def __call__(self, left: DataFrame, right: DataFrame) -> DataFrame:
        """Join two polars dataframes.

        Parameters
        ----------
        left: DataFrame
            Left dataframe in the join.
        right: DataFrame
            Right dataframe in the join.

        Returns
        -------
        DataFrame
            The joined dataframes.

        """
        return left.join(
            right,
            self.on,
            self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order
        )


class ToPandas(ArgRepr):
    """Partial of the polars dataframe `to_pandas <pandas_>`__ method.

    Parameters
    ----------
    use_pyarrow_extension_array: bool, optional
        Use pyarrow-backed extension arrays instead of numpy arrays for the
        columns of the pandas dataframe. This allows zero copy operations and
        preservation of null values. Subsequent operations on the resulting
        pandas dataframe may trigger conversion to numpy if those operations
        are not supported by pyarrow compute. Defaults to ``False``.
    **kwargs
        Additional keyword arguments to be passed to `pyarrow.Table.to_pandas()
        <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#
        pyarrow.Table.to_pandas>`_.


    .. _pandas: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.to_pandas.html

    """

    def __init__(
            self,
            use_pyarrow_extension_array: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(
            use_pyarrow_extension_array=use_pyarrow_extension_array,
            **kwargs
        )
        self.use_pyarrow_extension_array = use_pyarrow_extension_array
        self.kwargs = kwargs

    def __call__(self, df: DataFrame) -> PandasFrame:
        """Convert a polars dataframe into a pandas one.

        Parameters
        ----------
        df: DataFrame
            The polars dataframe to convert.

        Returns
        -------
        PandasFrame
            The converted pandas dataframe.

        """
        return df.to_pandas(
            use_pyarrow_extension_array=self.use_pyarrow_extension_array,
            **self.kwargs
        )

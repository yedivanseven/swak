from datetime import timedelta
from polars.dataframe.group_by import DynamicGroupBy
from polars import DataFrame
from polars._typing import IntoExpr, StartBy, ClosedInterval, Label
from ...misc import ArgRepr
from ..types import IntoExprs


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

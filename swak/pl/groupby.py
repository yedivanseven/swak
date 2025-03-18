from polars.dataframe.group_by import GroupBy, DynamicGroupBy
from polars import DataFrame
from polars._typing import IntoExpr
from ..misc import ArgRepr
from .types import IntoExprs


class GroupByAgg(ArgRepr):
    """Partial of a polars (dynamic) group-by object's  `agg <agg_>`__ method.

    Parameters
    ----------
    *aggs: IntoExpr
        Aggregations to compute for each group of the group by operation,
        specified as positional arguments. Accepts expression input.
        Strings are parsed as column names.
    **named_aggs: IntoExpr
        Additional aggregations, specified as keyword arguments. The resulting
        columns will be renamed to the keyword used.


    .. _agg: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
             polars.dataframe.group_by.GroupBy.agg.html#polars.dataframe.
             group_by.GroupBy.agg

    """

    def __init__(self, *aggs: IntoExprs, **named_aggs: IntoExpr) -> None:
        super().__init__(*aggs, **named_aggs)
        self.aggs = aggs
        self.named_aggs = named_aggs

    def __call__(self, grouped: GroupBy | DynamicGroupBy) -> DataFrame:
        """Aggregate a polars (dynamic) group-by object.

        Parameters
        ----------
        grouped: GroupBy or DynamicGroupBy
            The polars (dynamic) group-by object to aggregate.

        Returns
        -------
        DataFrame
            The aggregated (dynamic) group-by object.

        """
        return grouped.agg(*self.aggs, **self.named_aggs)

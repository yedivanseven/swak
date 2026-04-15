from polars.dataframe.group_by import GroupBy as GroupByT
from polars import DataFrame
from polars._typing import IntoExpr
from ...misc import ArgRepr
from ..types import IntoExprs


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
               polars.DataFrame.group_by.html#polars.DataFrame.group_by

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

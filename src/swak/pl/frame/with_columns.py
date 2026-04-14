from polars import DataFrame
from polars._typing import IntoExpr
from ...misc import ArgRepr
from ..types import IntoExprs


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

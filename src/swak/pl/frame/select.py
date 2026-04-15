from polars import DataFrame
from polars._typing import IntoExpr
from ...misc import ArgRepr
from ..types import IntoExprs


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
            The dataframe to select columns from.

        Returns
        -------
        DataFrame
            The selected columns.

        """
        return df.select(*self.exprs, **self.named_exprs)

from collections.abc import Iterable
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from ..misc import ArgRepr

type Column = ColumnNameOrSelector | Iterable[ColumnNameOrSelector]


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
              polars.DataFrame.drop.html#polars.DataFrame.drop

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
            The dataframe to drop columns from.

        Returns
        -------
        DataFrame
            The Ddtaframe without the dropped `columns`.

        """
        return df.drop(*self.columns, strict=self.strict)

from collections.abc import Collection
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from swak.misc import ArgRepr

type Subset = ColumnNameOrSelector | Collection[ColumnNameOrSelector]


class DropNulls(ArgRepr):
    """Partial of the polars dataframe `drop_nulls <drop_nulls_>`__ method.

    Parameters
    ----------
    subset: Subset
        Column name(s) for which null values are considered.
        If set to ``None`` (default), use all columns.


    .. _drop_nulls: https://docs.pola.rs/api/python/stable/reference/dataframe/
                    api/polars.DataFrame.drop_nulls.html

    """

    def __init__(self, subset: Subset | None = None) -> None:
        super().__init__(subset)
        self.subset = subset

    def __call__(self, df: DataFrame) -> DataFrame:
        """Drop rows from a polars dataframe where considered columns are null.

        Parameters
        ----------
        df: DataFrame
            The dataframe to to drop rows from.

        Returns
        -------
        DataFrame
            The dataframe with rows containing null values dropped.

        """
        return df.drop_nulls(self.subset)

from typing import Any
from collections.abc import Sequence
from polars import DataFrame, Series
from polars._typing import ColumnNameOrSelector
from swak.misc import ArgRepr


class Pivot(ArgRepr):
    """Partial of the polars dataframe `pivot <pivot_>`__ method.

    Parameters
    ----------
    on: ColumnNameOrSelector
        The column(s) whose values will be used as the new columns of the
        output dataframe.
    on_columns: Sequence or None
       What value combinations will be considered for the output table.


    .. _pivot: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.pivot.html#polars.DataFrame.pivot

    """

    def __init__(
            self,
            on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
            on_columns: Sequence[Any] | Series | DataFrame | None = None,
            **kwargs: Any
    ) -> None:
        super().__init__(on, on_columns, **kwargs)
        self.on = on
        self.on_columns = on_columns
        self.kwargs = kwargs

    def __call__(self, df: DataFrame) -> DataFrame:
        """Pivot a polars dataframe with the cached (keyword) arguments.

        Parameters
        ----------
        df: DataFrame
            The dataframe to pivot.

        Returns
        -------
        DataFrame
            The pivoted dataframe.

        """
        return df.pivot(self.on, self.on_columns, **self.kwargs)

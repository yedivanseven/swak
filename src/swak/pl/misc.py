from typing import overload
from polars._typing import SchemaDict
from ..misc import ArgRepr
from pandas import (
    Series as PandasSeries,
    DataFrame as PandasFrame,
    Index,
    DatetimeIndex
)
from polars import Series, DataFrame
import polars as pl


class FromPandas(ArgRepr):
    """Partial of the polars top-level function `from_pandas <pd_>`__.

    Parameters
    ----------
    schema_overrides: dict, optional
        Support override of inferred types for one or more columns.
        Defaults to ``None``.
    rechunk: bool, optional
        Make sure that all data is in contiguous memory. Default to ``True``.
    nan_to_null: bool, optional
        Pyarrow will convert the ``NaN`` to ``None``. Default to ``True``.
    include_index: bool, optional
        Load any non-default pandas indexes as columns. Default to ``False``.


    .. _pd: https://docs.pola.rs/api/python/stable/reference/api/
            polars.from_pandas.html

    """

    def __init__(
            self,
            schema_overrides: SchemaDict | None = None,
            rechunk: bool = True,
            nan_to_null: bool = True,
            include_index: bool = False,
    ) -> None:
        super().__init__(
            schema_overrides=schema_overrides,
            rechunk=rechunk,
            nan_to_null=nan_to_null,
            include_index=include_index
        )
        self.schema_overrides = schema_overrides
        self.rechunk = rechunk
        self.nan_to_null = nan_to_null
        self.include_index = include_index

    @overload
    def __call__(self, pandas: PandasSeries | Index | DatetimeIndex) -> Series:
        ...

    @overload
    def __call__(self, pandas: PandasFrame) -> DataFrame:
        ...

    def __call__(self, pandas):
        """Convert pandas structures into polars series or dataframes

        Parameters
        ----------
        pandas:
            Dataframe, series, or index to convert to polars.

        Returns
        -------
        Series or DataFrame
            Series if pandas series or index, dataframe otherwise.

        """
        return pl.from_pandas(
            pandas,
            schema_overrides=self.schema_overrides,
            rechunk=self.rechunk,
            nan_to_null=self.nan_to_null,
            include_index=self.include_index
        )

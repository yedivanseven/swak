from typing import overload
from collections.abc import Iterable
from ..misc import ArgRepr
from pandas import (
    Series as PandasSeries,
    DataFrame as PandasFrame,
    Index,
    DatetimeIndex
)
import polars as pl
from polars import Series, DataFrame
from polars._typing import (
    FrameInitTypes,
    SchemaDefinition,
    SchemaDict,
    Orientation,
    ConcatMethod,
    PolarsType
)


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


class Create(ArgRepr):
    """Partial of the polars top-level DataFrame constructor.

    Parameters
    ----------
    schema: SchemaDefinition
        The schema of the resulting DataFrame.
    schema_overrides: dict, optional
        Override of inferred types for one or more columns.
        Defaults to ``None``.
    strict: bool, optional
        Throw an error if any data value does not exactly match the given or
        inferred data type for that column. If set to ``False``, values that
        do not match the data type are cast to that data type or, if casting
        is not possible, set to null instead.
    orient: Orientation, optional
        Whether to interpret two-dimensional data as columns or as rows.
        Must be one of "row", "col", or ``None`` (the default).
    infer_schema_length: int, optional
        The maximum number of rows to scan for schema inference. If set to
        ``None``, the full data may be scanned (this can be slow).
        Defaults to 100.
    nan_to_null: bool, optional
        If the data comes from one or more numpy arrays, can optionally
        convert input data np.nan values to null instead.
        Defaults to ``False``.
    height: int, optional
        Allows constructing DataFrames with 0 width and a specified height.
        If passed with data, ensures the resulting DataFrame has this height.
        Defaults to ``None``.

    """

    def __init__(
            self,
            schema: SchemaDefinition | None = None,
            schema_overrides: SchemaDict | None = None,
            strict: bool = True,
            orient: Orientation | None = None,
            infer_schema_length: int | None = 100,
            nan_to_null: bool = False,
            height: int | None = None,
    ) -> None:
        super().__init__(
            schema,
            schema_overrides=schema_overrides,
            strict=strict,
            orient=orient,
            infer_schema_length=infer_schema_length,
            nan_to_null=nan_to_null,
            height=height
        )
        self.schema=schema
        self.schema_overrides = schema_overrides
        self.strict = strict
        self.orient = orient
        self.infer_schema_length = infer_schema_length
        self.nan_to_null = nan_to_null
        self.height = height

    def __call__(self, data: FrameInitTypes) -> DataFrame:
        """Create a polars dataframe with the cached parameters.

        Parameters
        ----------
        data: FrameInitTypes
            The data to create the polars dataframe from.

        Returns
        -------
        DataFrame
            A polars DataFrame.

        """
        return DataFrame(
            data,
            self.schema,
            schema_overrides=self.schema_overrides,
            strict=self.strict,
            orient=self.orient,
            infer_schema_length=self.infer_schema_length,
            nan_to_null=self.nan_to_null,
            height=self.height
        )


class Concat(ArgRepr):
    """Partial of the polars top-level `concat` function.

    Parameters
    ----------
    how: str, optional
        One of "vertical", "vertical_relaxed", "diagonal", "diagonal_relaxed",
        "horizontal", "align", "align_full", "align_inner", "align_left",
        "align_right". Defaults to "vertical".
    rechunk: bool, optional
        Make sure that the result data is in contiguous memory.
        Defaults to ``False``.
    parallel: bool, optional
        Only relevant for LazyFrames. Defaults to ``True``.
    strict: bool, optional
        When `how` is "horizontal", require all DataFrames to be the same
        height, raising an error if not. Defaults to ``False``.

    """

    def __init__(
            self,
            how: ConcatMethod = 'vertical',
            rechunk: bool = False,
            parallel: bool = True,
            strict: bool = False
    ) -> None:
        super().__init__(
            how=how,
            rechunk=rechunk,
            parallel=parallel,
            strict=strict
        )
        self.how=how
        self.rechunk=rechunk
        self.parallel = parallel
        self.strict = strict

    def __call__(self, items: Iterable[PolarsType]) -> PolarsType:
        """Concatenate polars dataframes respecting with the cached options.

        Parameters
        ----------
        items: iterable of PolarsType
            The dataframes to concatenate.

        Returns
        -------
        PolarsType
            The concatenated dataframes.

        """
        return pl.concat(
            items,
            how=self.how,
            rechunk=self.rechunk,
            parallel=self.parallel,
            strict=self.strict,
        )

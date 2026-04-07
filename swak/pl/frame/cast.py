from collections.abc import Mapping
from polars import DataFrame, Schema
from polars._typing import ColumnNameOrSelector, PolarsDataType, PythonDataType
from ...misc import ArgRepr

type DataType = PolarsDataType | PythonDataType
type Dtypes = (
    Mapping[ColumnNameOrSelector | PolarsDataType, DataType]
    | PolarsDataType
    | Schema
)


class Cast(ArgRepr):
    """Partial of the polars dataframe `cast <cast_>`__ method.

    Parameters
    ----------
    dtypes: Dtypes
        Mapping of column names (or selector) to dtypes, or a single dtype
        to which all columns will be cast.
    strict: bool, optional
        Raise if cast is invalid on rows after predicates are pushed down.
        If ``False``, invalid casts will produce null values.
        Defaults to ``True``


    .. _cast: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.cast.html#polars.DataFrame.cast

    """

    def __init__(self, dtypes: Dtypes, *, strict: bool = True) -> None:
        super().__init__(dtypes, strict=strict)
        self.dtypes = dtypes
        self.strict = strict

    def __call__(self, df: DataFrame) -> DataFrame:
        """Cast column(s) of a polars dataframe to the cached type(s).

        Parameters
        ----------
        df: DataFrame
            The dataframe to type-cast column(s) of.

        Returns
        -------
        DataFrame
            The dataframe with its column(s) cast to the new type(s).

        """
        return df.cast(self.dtypes, strict=self.strict)

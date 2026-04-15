from typing import Any
from pandas import DataFrame as PandasFrame
from polars import DataFrame
from ...misc import ArgRepr


class ToPandas(ArgRepr):
    """Partial of the polars dataframe `to_pandas <pandas_>`__ method.

    Parameters
    ----------
    use_pyarrow_extension_array: bool, optional
        Use pyarrow-backed extension arrays instead of numpy arrays for the
        columns of the pandas dataframe. This allows zero copy operations and
        preservation of null values. Subsequent operations on the resulting
        pandas dataframe may trigger conversion to numpy if those operations
        are not supported by pyarrow compute. Defaults to ``False``.
    **kwargs
        Additional keyword arguments to be passed to `pyarrow.Table.to_pandas()
        <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#
        pyarrow.Table.to_pandas>`_.


    .. _pandas: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.to_pandas.html

    """

    def __init__(
            self,
            use_pyarrow_extension_array: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(
            use_pyarrow_extension_array=use_pyarrow_extension_array,
            **kwargs
        )
        self.use_pyarrow_extension_array = use_pyarrow_extension_array
        self.kwargs = kwargs

    def __call__(self, df: DataFrame) -> PandasFrame:
        """Convert a polars dataframe into a pandas one.

        Parameters
        ----------
        df: DataFrame
            The polars dataframe to convert.

        Returns
        -------
        PandasFrame
            The converted pandas dataframe.

        """
        return df.to_pandas(
            use_pyarrow_extension_array=self.use_pyarrow_extension_array,
            **self.kwargs
        )

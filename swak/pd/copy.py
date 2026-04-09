from typing import overload
from pandas import DataFrame, Series
from ..misc import ArgRepr


class Copy(ArgRepr):
    """Partial of the ``copy`` method of a pandas dataframe or series.

    Parameters
    ----------
    deep: bool, optional
        Makes a deep copy when ``True``, including a copy of the data and the
        indices. When ``False``, neither the indices nor the data are copied.
        Defaults to ``True``

    """

    def __init__(self, deep: bool = True) -> None:
        self.deep = bool(deep)
        super().__init__(self.deep)


    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    def __call__(self, df):
        """Call the ``copy`` method of the passed pandas dataframe or series.

        Parameters
        ----------
        df: DataFrame or Series
            The pandas object to call the ``copy`` method of.

        Returns
        -------
        DataFrame or Series
            Copy of the pandas object passed, deep or shallow, depending on
            the flag set at instantiation.

        """
        return df.copy(self.deep)

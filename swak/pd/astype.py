from typing import Any, overload
from collections.abc import Hashable, Mapping
import numpy as np
from pandas.core.dtypes.base import ExtensionDtype
from pandas import DataFrame, Series
from ..misc.repr import ReprName

type Type = str | type | np.dtype | ExtensionDtype
type Types = Mapping[Hashable, Type]


class AsType(ReprName):
    """Partial of a pandas dataframe or series ``astype`` method.

    Parameters
    ----------
    types: type or dict
        Single type or dictionary of column names and types, specifying type
        conversion of entire dataframe or specific columns, respectively
    **kwargs
        Keyword arguments are passed on to the ``astype`` method call after
        the `types` argument.

    """
    def __init__(self, types: Type | Types, **kwargs: Any) -> None:
        super().__init__()
        self.types = types
        self.kwargs = kwargs

    def __repr__(self) -> str:
        kwargs = (f'{k}={self._repr(v)}' for k, v in self.kwargs.items())
        kwargs = ', '.join(kwargs)
        try:
            args = (f'{k!r}: {self._name(self.types[k])}' for k in self.types)
            args = '{' + ', '.join(args) + '}'
        except TypeError:
            args = self._name(self.types)
        signature = ', '.join(filter(None, [args, kwargs]))
        return f'{self.__class__.__name__}({signature})'

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Cast dataframe (columns) or series to specified types.

        Parameters
        ----------
        df: DataFrame or Series
            Pandas dataframe tor series o type-cast.

        Returns
        -------
        DataFrame or Series
            Pandas dataframe (columns) or series cast to new type(s).

        """
        return df.astype(self.types, **self.kwargs)

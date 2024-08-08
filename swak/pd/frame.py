from typing import Any, Hashable, Iterable, Callable
from numpy import dtype, ndarray
from pandas.core.dtypes.base import ExtensionDtype
from pandas import DataFrame, Series
from ..magic.repr import ReprName
from ..magic import ArgRepr

type Type = str | type | dtype | ExtensionDtype
type Types = dict[Hashable, Type]
type Mask = list[bool] | Series | ndarray
type Condition = Callable[[DataFrame], Mask]


class AsType(ReprName):
    """Partial of the ``pandas.DataFrame.astype`` method.

    Parameters
    ----------
    types: type or dict
        Single type or dictionary of column names and types, specifying type
        conversion of entire DataFrame or specific columns, respectively
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

    def __call__(self, df: DataFrame) -> DataFrame:
        """Cast entire dataframe or columns thereof to specified types

        Parameters
        ----------
        df: DataFrame
            Pandas dataframe to type-cast.

        Returns
        -------
        DataFrame
            Pandas dataframe cast to new type or with columns cast to new types.

        """
        return df.astype(self.types, **self.kwargs)


class ColumnSelector(ArgRepr):
    """Select a single column of a pandas dataframe as a pandas series.

    This is simply a partial for calling a dataframe's ``__getitem__``
    method with a single argument (using the square-brackets accessor).

    Parameters
    ----------
    col: hashable
        Single DataFrame column to select.

    """

    def __init__(self, col: Hashable) -> None:
        self.col = self.__valid(col)
        super().__init__(col)

    def __call__(self, df: DataFrame) -> Series:
        """

        Parameters
        ----------
        df: DataFrame
            Pandas dataframe to select column from.

        Returns
        -------
        Series
            The selected dataframe column.

        """
        return df[self.col]

    @staticmethod
    def __valid(col: Hashable) -> Hashable:
        _ = hash(col)
        return col


class ColumnsSelector(ArgRepr):
    """Select one or more columns of a pandas dataframe as dataframe.

    This is simply a partial for calling a dataframe's ``__getitem__``
    method with a list of arguments (using the square-brackets accessor).

    Parameters
    ----------
    col: Hashable, optional
        Column name or iterable thereof. Defaults to an empty tuple.
    *cols: Hashable
        Additional columns names.

    """

    def __init__(
            self,
            col: Hashable | Iterable[Hashable] = (),
            *cols: Hashable
    ) -> None:
        self.cols: tuple[Hashable, ...] = self.__valid(col) + self.__valid(cols)
        super().__init__(*self.cols)

    def __call__(self, df: DataFrame) -> DataFrame:
        """Select the specified column(s) from a pandas DataFrame.

        Parameters
        ----------
        df: DataFrame
            Pandas dataframe to select column(s) from.

        Returns
        -------
        DataFrame
            The selected dataframe column(s).

        """
        return df[list(self.cols)]

    @staticmethod
    def __valid(cols: Hashable | Iterable[Hashable]) -> tuple[Hashable, ...]:
        """Ensure that the columns are indeed an iterable of hashables."""
        if isinstance(cols, str):
            return cols,
        try:
            _ = [hash(col) for col in cols]
        except TypeError:
            _ = hash(cols)
            return cols,
        return tuple(cols)


class RowsSelector(ArgRepr):
    """Select rows from a pandas dataframe with a boolean mask or function.

    This is simply a partial for calling a dataframe's ``__getitem__``
    method (using the square-brackets accessor) with a 1-D, boolean array-like
    structure (of the same length as the dataframe to select from) or callable
    that produces such a boolean mask from the dataframe.

    Parameters
    ----------
    condition: callable or array-like
        Either a 1-D, array-like structure of boolean (e.g., a list, a numpy
        array, or a pandas series, but not a tuple!) or a callable that accepts
        a pandas dataframe and produces such an array-like structure.

    """

    def __init__(self, condition: Mask | Condition) -> None:
        super().__init__(condition)
        self.condition = condition

    def __call__(self, df: DataFrame) -> DataFrame:
        """Select rows from a dataframe with the specified mask or condition.

        Parameters
        ----------
        df: DataFrame
            The pandas dataframe to select rows from.

        Returns
        -------
        DataFrame
            The pandas dataframe with only the selected rows.

        """
        return df[self.condition]

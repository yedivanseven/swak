from typing import Any
from collections.abc import Hashable, Iterable, Callable
from numpy import dtype, ndarray
from pandas.core.dtypes.base import ExtensionDtype
from pandas import DataFrame, Series
from ..misc.repr import ReprName
from ..misc import ArgRepr

type Type = str | type | dtype | ExtensionDtype
type Types = dict[Hashable, Type]
type Mask = list[bool] | Series | ndarray
type Condition = Callable[[DataFrame], Mask]
type Transform = dict[Hashable, Any] | Series | Callable[[Any], Any]


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
            Pandas dataframe cast to new type or with columns cast to
            new types.

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
        col = self.__valid(col)
        self.cols: tuple[Hashable, ...] = col + self.__valid(cols)
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


class ColumnMapper(ArgRepr):
    """Transform one column of a pandas dataframe into another.

    This is simply a partial of calling the ``map`` method on one column of a
    dataframe and assigning the result to the same or another column of the
    same dataframe.

    Parameters
    ----------
    src_col: Hashable
        Column to call the ``map`` method on.
    transform: callable, Mapping, or Series
        Function or mapping in the form of a dictionary or a pandas series.
    tgt_col: Hashable, optional
        Dataframe column to store the series resulting from the
        transformation. Defaults to `src_col`, thus overwriting it in place.
    **kwargs
        Keyword arguments are passed on to the call of the Series'
        ``map`` method.

    """

    def __init__(
            self,
            src_col: Hashable,
            transform: Transform,
            tgt_col: Hashable | None = None,
            **kwargs: Any
    ) -> None:
        self.src_col = src_col
        self.tgt_col = src_col if tgt_col is None else tgt_col
        self.transform = transform
        self.kwargs = kwargs
        name = transform if callable(transform) else type(transform)
        super().__init__(src_col, name, tgt_col, **kwargs)

    def __call__(self, df: DataFrame) -> DataFrame:
        """Called the ``map`` method on a specified column of a DataFrame.

        Cached keyword arguments are forwarded to the method call and
        the result is stored in the specified column of the DataFrame.

        Parameters
        ----------
        df: DataFrame
            Pandas dataframe with the column to call the ``map`` method on.

        Returns
        -------
        DataFrame
            Pandas dataframe with the result of the column transformation in
            the specified column.


        """
        df[self.tgt_col] = df[self.src_col].map(self.transform, **self.kwargs)
        return df


class RowsSelector(ArgRepr):
    """Select rows from a pandas dataframe with a boolean mask or function.

    This is simply a partial for calling a dataframe's ``__getitem__``
    method (using the square-brackets accessor) with a callable that takes
    the dataframe as input, and produces a 1-D, boolean array-like structure
    (of the same length as the dataframe to select from).

    Parameters
    ----------
    condition: callable or array-like
        A callable that accepts a dataframe and produces a 1-D, boolean array-
        like structure of the same length

    """

    def __init__(self, condition: Condition) -> None:
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

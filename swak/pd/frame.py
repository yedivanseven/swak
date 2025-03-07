from typing import Any
from collections.abc import Hashable, Iterable, Callable
from functools import singledispatchmethod
from numpy import dtype, ndarray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series
from ..misc.repr import ReprName
from ..misc import ArgRepr

type Type = str | type | dtype | ExtensionDtype
type Types = dict[Hashable, Type]
type Mask = list[bool] | Series | ndarray
type Condition = Callable[[DataFrame], Mask]
type Transform = dict[Hashable, Any] | Series | Callable[[Any], Any]
type Others = Series | DataFrame | list[Series | DataFrame]


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
    """Select a single column of a (grouped) pandas dataframe as a series.

    This is simply a partial for calling a (grouped) dataframe's
    ``__getitem__`` method with a single argument (using the square-brackets
    accessor).

    Parameters
    ----------
    col: hashable
        Single DataFrame column to select.

    """

    def __init__(self, col: Hashable) -> None:
        self.col = self.__valid(col)
        super().__init__(col)

    @singledispatchmethod
    def __call__(self, df: DataFrame) -> Series:
        """Select a single column of a (grouped) pandas dataframe as series.

        Parameters
        ----------
        df: DataFrame or DataFrameGroupBy
            Pandas dataframe or grouped dataframe to select column from.

        Returns
        -------
        Series or SeriesGroupBy
            The selected column from the (grouped) dataframe.

        """
        return df[self.col]

    @__call__.register
    def _(self, grouped_df: DataFrameGroupBy) -> SeriesGroupBy:
        return grouped_df[self.col]

    @staticmethod
    def __valid(col: Hashable) -> Hashable:
        _ = hash(col)
        return col


class ColumnsSelector(ArgRepr):
    """Select one or more columns of a (grouped) pandas dataframe as dataframe.

    This is simply a partial for calling a (grouped) dataframe's
    ``__getitem__``  method with a list of arguments (using the square-brackets
    accessor).

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

    @singledispatchmethod
    def __call__(self, df: DataFrame | DataFrameGroupBy) -> DataFrame:
        """Select the specified column(s) from a (grouped) pandas dataframe.

        Parameters
        ----------
        df: DataFrame or DataFrameGroupBy
            Pandas dataframe or grouped dataframe to select column(s) from.

        Returns
        -------
        DataFrame or DataFrameGroupBy
            The selected column(s) of the (grouped) dataframe.

        """
        return df[list(self.cols)]

    @__call__.register
    def _(self, grouped_df: DataFrameGroupBy) -> DataFrameGroupBy:
        return grouped_df[list(self.cols)]

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


class FrameGroupBy(ArgRepr):
    """Simple partial of a pandas dataframe's ``groupby`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``groupby`` method call.
    **kwargs
        Keyword arguments to pass on to the ``groupby`` method call.

    Note
    ----
    For a full list of (keyword) arguments and their description, see the
    pandas `documentation <https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.groupby.html>`_.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df: DataFrame) -> DataFrameGroupBy:
        """Call the dataframe's ``groupby`` method with the cached (kw)args.

        Parameters
        ----------
        df: DataFrame
            Pandas dataframe to group.

        Returns
        -------
        DataFrameGroupBy
            The grouped dataframe.

        """
        return df.groupby(*self.args, **self.kwargs)


class Join(ArgRepr):
    """Light wrapper around the pandas dataframe ``join`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``join`` method call.
    **kwargs
        Keyword arguments to pass on to the ``join`` method call.

    Note
    ----
    For a full list of (keyword) arguments and their description, see the
    pandas `join documentation <https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.join.html>`_.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, df: DataFrame, other: Others) -> DataFrame:
        """Join a dataframe with other dataframe(s) and/or series.

        Parameters
        ----------
        df: DataFrame
            Source dataframe on which the ``join`` method will be called.
        other: DataFrame, Series, or a list of any combination
            Index should be similar to one (or more) columns in `df`. If a
            series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined dataframe.

        Returns
        -------
        DataFrame
            The joined dataframe.

        """
        return df.join(other, *self.args, **self.kwargs)


class Assign(ArgRepr):
    """Light wrapper around a pandas dataframe's ``assign`` method.

    Parameters
    ----------
    col: dict
        A dictionary with the names of newly created (or overwritten) columns
        as keys. If the values are callable, they are computed on the entire
        dataframe and assigned to the new columns. The callable must not
        the change input dataframe (though pandas doesn’t check it). If the
        values are not callable, e.g., a series, scalar, or array, they are
        simply assigned.
    **cols
        As in the original, the keyword arguments themselves serve as the
        name(s) of the new (or overwritten) column(s) and their values are set
        in the same way.

    """

    def __init__(self, col: dict[str, Any] | None = None, **cols: Any) -> None:
        self.cols = cols if col is None else col | cols
        super().__init__(**self.cols)

    def __call__(self, df: DataFrame) -> DataFrame:
        """Add new columns to a dataframe by calling its ``assign`` method.

        Parameters
        ----------
        df: DataFrame
            The dataframe to add new columns to.

        Returns
        -------
        DataFrame
            The input dataframe with new columns added.

        """
        return df.assign(**self.cols)

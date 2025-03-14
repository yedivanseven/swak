from typing import Literal, Any, overload
from collections.abc import Hashable, Callable, Sequence
from numpy import dtype, ndarray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series
from ..misc.repr import ReprName
from ..misc import ArgRepr

type Type = str | type | dtype | ExtensionDtype
type Types = dict[Hashable, Type]
type Mask = list[bool] | Series | ndarray[bool]
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

    @overload
    def __call__(self, df: DataFrame) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> SeriesGroupBy:
        ...

    def __call__(self, df):
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
        Column name or sequence thereof. Defaults to an empty tuple.
    *cols: Hashable
        Additional columns names.

    """

    def __init__(
            self,
            col: Hashable | Sequence[Hashable] = (),
            *cols: Hashable
    ) -> None:
        col = self.__valid(col)
        self.cols: tuple[Hashable, ...] = col + self.__valid(cols)
        super().__init__(*self.cols)

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> DataFrameGroupBy:
        ...

    def __call__(self, df):
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

    @staticmethod
    def __valid(cols: Hashable | Sequence[Hashable]) -> tuple[Hashable, ...]:
        """Ensure that the columns are indeed a sequence of hashables."""
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
    na_action: str, optional
        Can take the value "ignore" or ``None``, defaulting to the latter.
        Will be passed to the series ``map`` method along with `transform`.

    """

    def __init__(
            self,
            src_col: Hashable,
            transform: Transform,
            tgt_col: Hashable | None = None,
            na_action: Literal['ignore'] | None = None
    ) -> None:
        self.src_col = src_col
        self.tgt_col = src_col if tgt_col is None else tgt_col
        self.transform = transform
        self.na_action = na_action
        name = transform if callable(transform) else type(transform)
        super().__init__(src_col, name, tgt_col, na_action=na_action)

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
        df[self.tgt_col] = df[self.src_col].map(self.transform, self.na_action)
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


class Drop(ArgRepr):
    """A simple partial of a dataframe's or series' ``drop`` method.

    Parameters
    ----------
    labels: hashable or sequence, optional
        Index or column labels to drop. Defaults to ``None``.
    axis: 1 or "columns", 0 or "index"
        Whether to drop labels from the columns (1 or "columns") or
        index (0 or "index"). Defaults to 1
    index: hashable or sequence, optional
        Single label or list-like. Defaults to ``None``.  Alternative to
        specifying axis (labels, axis=0 is equivalent to index=labels).
    columns: hashable or sequence, optional
        Single label or list-like. Defaults to ``None``. Alternative to
        specifying axis (labels, axis=1 is equivalent to columns=labels).
    level: hashable, optional.
        Integer or level name. Defaults to ``None``. For MultiIndex, level
        from which the labels will be removed.
    errors: "raise" or "ignore"
        Defaults to "raise". If "ignore", suppress error and drop only
        existing labels.

    """

    def __init__(
            self,
            label: Hashable | Sequence[Hashable] | None = None,
            *labels: Hashable,
            axis: int | Literal['index', 'columns', 'rows'] = 1,
            index: Hashable | Sequence[Hashable] | None = None,
            columns: Hashable | Sequence[Hashable] | None = None,
            level: Hashable | None = None,
            errors: Literal['ignore', 'raise'] = 'raise'
    ) -> None:
        self.labels = (self.__valid(label) + self.__valid(labels)) or None
        self.axis = axis
        self.index = index
        self.columns = columns
        self.level = level
        self.errors = errors
        super().__init__(
            self.labels,
            axis= axis,
            index=index,
            columns=columns,
            level=level,
            errors=errors
        )

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Drop rows or columns from a pandas series or dataframe.

        Parameters
        ----------
        df: Series or DataFrame
            The object to drop rows or columns from.

        Returns
        -------
        Series or DataFrame
            The object with rows or columns dropped.

        """
        return df.drop(
            self.labels,
            axis=self.axis,
            index=self.index,
            columns=self.columns,
            level=self.level,
            inplace=False,
            errors=self.errors,
        )

    @staticmethod
    def __valid(labels: Hashable | Sequence[Hashable]) -> list[Hashable]:
        """Ensure that the labels are indeed a sequence of hashables."""
        if labels is None:
            return []
        if isinstance(labels, str):
            return [labels]
        try:
            _ = [hash(label) for label in labels]
        except TypeError:
            _ = hash(labels)
            return [labels]
        return list(labels)


class DropNA(ArgRepr):
    """A simple partial of a dataframe's or series' ``dropna`` method.

    Parameters
    ----------
    axis: 0 or "index", 1 or "columns"
        Determine if rows or columns which contain missing values are removed.
        Defaults to 0.
    how: "any" or "all"
        Determine if row or column is removed from DataFrame, when we have at
        least one NA or all NA. Defaults to "any".
    thresh: int, optional
        Require that many non-NA values. Cannot be combined with how.
        Defaults to ``None``.
    subset: hashable or sequence, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include. Defaults to ``None``.
    ignore_index: bool
        Defaults to ``False``. If True, the resulting axis will be labeled
        0, 1, …, n - 1.

    """

    def __init__(
            self,
            axis: int | Literal['index', 'columns', 'rows'] = 0,
            how: Literal['any', 'all'] | None = None,
            thresh: int | None = None,
            subset: Hashable | Sequence[Hashable] | None = None,
            ignore_index: bool = False
    ) -> None:
        self.axis = axis
        self.how = how
        self.thresh = thresh
        self.subset = subset
        self.ignore_index = ignore_index
        super().__init__(
            axis=self.axis,
            how=self.how,
            thresh=self.thresh,
            subset=self.subset,
            ignore_index=self.ignore_index
        )

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Drop rows or columns with NAs from a pandas series or dataframe.

        Parameters
        ----------
        df: Series or DataFrame
            The object to drop rows or columns with NAs from.

        Returns
        -------
        Series or DataFrame
            The object with rows or columns with NAs dropped.

        """
        return df.dropna(
            axis=self.axis,
            **({'how': self.how} if self.how else {'thresh': self.thresh}),
            subset=self.subset,
            inplace=False,
            ignore_index=self.ignore_index
        )


class SortValues(ArgRepr):
    """Partial of the pandas dataframe ``sort_values`` method.

    Parameters
    ----------
    by: hashable or sequence
        Name or list of names to sort by.
    **kwargs
        Additional keyword arguments will be forwarded to the method call with
        the exception of "inplace", which will be set to ``False``.

    Note
    ----
    For a full list of keyword arguments and their description, see the
    pandas `sort_values documentation <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.DataFrame.sort_values.html>`_.

    """

    def __init__(
            self,
            by: Hashable | Sequence[Hashable],
            **kwargs: Any
    ) -> None:
        super().__init__(by, **kwargs)
        self.by = by
        self.kwargs = (kwargs.pop('inplace', ''), kwargs)[1]

    def __call__(self, df: DataFrame) -> DataFrame:
        """Sort a pandas dataframe by column(s) values.

        Parameters
        ----------
        df: DataFrame
            The dataframe to sort.

        Returns
        -------
        DataFrame
            The sorted dataframe.

        """
        return df.sort_values(self.by, inplace=False, **self.kwargs)

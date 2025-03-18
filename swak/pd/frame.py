from typing import Literal, Any, overload
from collections.abc import Hashable, Callable, Sequence, Iterator, Mapping
from numpy import dtype, ndarray
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.window import Rolling, Window, RollingGroupby
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
from pandas import DataFrame, Series, Index, Grouper
from ..misc.repr import ReprName
from ..misc import ArgRepr

type Type = str | type | dtype | ExtensionDtype
type Types = dict[Hashable, Type]
type Mask = list[bool] | Series | ndarray[bool]
type Condition = Callable[[DataFrame], Mask]
type Transform = dict[Hashable, Any] | Series | Callable[[Any], Any]
type Others = Series | DataFrame | list[Series | DataFrame]
type Key = Hashable | ndarray[Hashable] | Series | Index | Iterator[Hashable]
type Keys = list[Key]
type Renamer = Mapping[Hashable, Hashable] | Callable[[Hashable], Hashable]
type GroupKey = (
    str
    | Callable[[Hashable], Hashable]
    | Grouper
    | Mapping[Hashable, Hashable]
    | ndarray[Hashable]
    | Series
)
type GroupKeys = list[GroupKey]
type Func = Callable[[Series], float] | str
type Funcs = list[Func] | dict[str, Func] | dict[str, list[Func]]


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
        A callable that accepts a dataframe and produces a 1-D, boolean
        array-like structure of the same length

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


class GroupBy(ArgRepr):
    """Simple partial of a pandas dataframe and series ``groupby`` method.

    Parameters
    ----------
    by: str, callable, series, array, dict, or list
        Column name, function (to be called on each column name), list or numpy
        array of the same length as the columns, a dict or series providing a
        label -> group name mapping, or a list of the above.
    level: hashable or sequence, optional
        If the axis is a multi-index (hierarchical), group by a particular
        level or levels. Do not specify both `by` and `level`.
        Defaults to ``None``.
    as_index: bool, optional
        Whether to return group labels as index. Defaults to ``True``.
    sort: bool, optional
        Whether to sort group keys. Defaults to ``True``.
    group_keys: bool, optional
        Defaults to ``True``
    observed: bool, optional
        Whether to show only observed values for categorical groupers.
        Defaults to ``False``.
    dropna: bool, optional
        Whether to treat NA values in group keys as groups.
        Defaults to ``True``.

    Note
    ----
    For a more extensive description of all (keyword) arguments, see the
    pandas `documentation <https://pandas.pydata.org/pandas-docs/stable/
    reference/api/pandas.DataFrame.groupby.html>`_.

    """

    def __init__(
            self,
            by: GroupKey | GroupKeys | None = None,
            level: Hashable | Sequence[Hashable] | None = None,
            as_index: bool = True,
            sort: bool = True,
            group_keys: bool = True,
            observed: bool = False,
            dropna: bool = True,
    ) -> None:
        super().__init__(
            by,
            level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna
        )
        self.by = by
        self.level = level
        self.as_index = as_index
        self.sort = sort
        self.group_keys = group_keys
        self.observed = observed
        self.dropna = dropna

    @overload
    def __call__(self, df: Series) -> SeriesGroupBy:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrameGroupBy:
        ...

    def __call__(self, df):
        """Call a dataframe or series  ``groupby`` method.

        Parameters
        ----------
        df: DataFrame or Series
            Pandas dataframe or series to group.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            The grouped dataframe or series.

        """
        return df.groupby(
            self.by,
            level=self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            observed=self.observed,
            dropna=self.dropna
        )


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
    """A simple partial of a pandas dataframe or series' ``drop`` method.

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
    """A simple partial of a pandas dataframe or series' ``dropna`` method.

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
    ignore_index: bool, optional
        Defaults to ``False``. If ``True``, the resulting axis will be labeled
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


class SetIndex(ArgRepr):
    """Simple partial of a pandas dataframe's ``set_index`` method.

    Parameters
    ----------
    keys: hashable or array-like
        This parameter can be either a single column key, a single array of
        the same length as the calling DataFrame, or a list containing an
        arbitrary combination of column keys and arrays.
    drop : bool, optional
        Delete columns to be used as the new index. Defaults to ``True``.
    append : bool, optional
        Whether to append columns to existing index. Defaults to ``False``
    verify_integrity : bool, optional
        Whether to check the new index for duplicates. Defaults to ``False``.
        Setting to ``True`` will impact the performance of this method.

    """

    def __init__(
            self,
            keys: Key | Keys,
            drop: bool = True,
            append: bool = False,
            verify_integrity: bool = False
    ) -> None:
        self.keys = keys
        self.drop = drop
        self.append = append
        self.verify_integrity = verify_integrity
        super().__init__(
            keys,
            drop=drop,
            append=append,
            verify_integrity=verify_integrity
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        """Set the index of a pandas dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to set the index of.

        Returns
        -------
        DataFrame
            The Dataframe with a new index set.

        """
        return df.set_index(
            self.keys,
            drop=self.drop,
            append=self.append,
            inplace=False,
            verify_integrity=self.verify_integrity
        )


class ResetIndex(ArgRepr):
    """Simple partial of a pandas dataframe's ``reset_index`` method.

    Parameters
    ----------
    level: int, str, tuple, or list, optional
        Only remove the given levels from the index. Defaults to ``None``,
        which removes all levels.
    drop: bool, optional
        Do not try to insert index into dataframe columns. This resets
        the index to the default integer index. Default to ``False``.
    col_level: int or str, optional
        If the columns have multiple levels, determines which level the
        labels are inserted into. Default to 0.
    col_fill: Hashable, optional
        If the columns have multiple levels, determines how the other
        levels are named. Defaults to an empty string.
    allow_duplicates : bool, optional
        Allow duplicate column labels to be created. Defaults to ``False``
    names : hashable or sequence, optional
        Using the given string, rename the dataframe column which contains the
        index data. If the dataframe has a multiindex, this has to be a list or
        tuple with length equal to the number of levels. Defaults to ``None``.

    """

    def __init__(
            self,
            level: Hashable | Sequence[Hashable] | None = None,
            drop: bool = False,
            col_level: Hashable = 0,
            col_fill: Hashable = '',
            allow_duplicates: bool = False,
            names: Hashable | Sequence[Hashable] | None = None,
    ) -> None:
        self.level = level
        self.drop = drop
        self.col_level = col_level
        self.col_fill = col_fill
        self.allow_duplicates = allow_duplicates
        self.names = names
        super().__init__(
            level,
            drop=drop,
            col_level=col_level,
            col_fill=col_fill,
            allow_duplicates=allow_duplicates,
            names=names
        )

    def __call__(self, df: DataFrame) -> DataFrame:
        """Reset the index of a pandas dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to reset the index of.

        Returns
        -------
        DataFrame
            The dataframe with its index reset.

        """
        return df.reset_index(
            self.level,
            drop=self.drop,
            inplace=False,
            col_level=self.col_level,
            col_fill=self.col_fill,
            allow_duplicates=self.allow_duplicates,
            names=self.names,
        )


class Rename(ArgRepr):
    """Simple partial of a pandas dataframe's ``rename`` method.

    Parameters
    ----------
    mapper : dict-like or function
        Dict-like or function transformations to apply to the `axis` values.
    index : dict-like or function
        Alternative to specifying `mapper` with `axis` = 0.
    columns : dict-like or function
        Alternative to specifying `mapper` with `axis` = 1.
    axis : 1 or "columns", 0 or "index", optional
        Axis to target with `mapper`. Defaults to 1.
    level : Hashable, optional
        In case of a MultiIndex, only rename labels in the specified level.
        Defaults to ``None``
    errors : "ignore" or "raise", optional
        If "raise", raise a ``KeyError`` when a dict-like `mapper`, `index`,
        or `columns` contains labels that are not present in the index
        being transformed. If "ignore", existing keys will be renamed and
        extra keys will be ignored. Defaults to "ignore".

    """

    def __init__(
            self,
            mapper: Renamer | None = None,
            index: Renamer | None = None,
            columns: Renamer | None = None,
            axis: int | Literal['index', 'columns', 'rows'] = 1,
            level: Hashable | None = None,
            errors: Literal['ignore', 'raise'] = 'ignore'
    ) -> None:
        self.mapper = mapper
        self.index = index
        self.columns = columns
        self.axis = axis
        self.level = level
        self.errors = errors
        super().__init__(
            mapper,
            index=self.index,
            columns=self.columns,
            axis=self.axis,
            level=self.level,
            errors=self.errors
        )

    @property
    def resolved(self) -> dict[str, Any]:
        """Resolved mapper-axis vs. index vs. columns keywords."""
        return {
            'index': self.index,
            'columns': self.columns
        } if self.mapper is None else {
            'axis': self.axis
        }



    def __call__(self, df: DataFrame) -> DataFrame:
        """Rename a pandas dataframe's columns or rows.

        Parameters
        ----------
        df: DataFrame
            The dataframe to rename columns or rows of.

        Returns
        -------
        DataFrame
            The dataframe with renamed columns or rows.

        """

        return df.rename(
            self.mapper,
            **self.resolved,
            level=self.level,
            inplace=False,
            errors=self.errors
        )


class Agg(ArgRepr):
    """Simple partial for calling a pandas object's ``agg`` method.

    Parameters
    ----------
    func: callable, str, list, or dict, optional
        Function(s) to use for aggregating the data. If a function, must work
        when passed a Series. Also acceptable are a function name, a list of
        function names and a dictionary with columns names as keys and
        functions, function names, or lists thereof as values.
        Defaults to ``None``, which only works for a dataframe and relies on
        `kwargs` to specify `named aggregations <https://pandas.pydata.org/
        pandas-docs/stable/user_guide/groupby.html#groupby-aggregate-named>`_.
    *args
        Positional arguments to pass on to the ``agg`` or `func` call.
    **kwargs
        Keyword arguments to pass on to the ``agg`` or `func` call.

    Note
    ----
    See the pandas `agg docs <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html>`_ for
    a full list of (keyword) arguments and an extensive description of
    usage and configuration.

    """

    def __init__(
            self,
            func: Func | Funcs | None = None,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(func, *args, **kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    @overload
    def __call__(self, df: DataFrame) -> Series | DataFrame:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> DataFrame:
        ...

    @overload
    def __call__(self, df: Series) -> Any:
        ...

    @overload
    def __call__(self, df: SeriesGroupBy) -> Series:
        ...

    @overload
    def __call__(self, df: Rolling) -> Series | DataFrame:
        ...

    @overload
    def __call__(self, df: RollingGroupby) -> Series | DataFrame:
        ...

    def __call__(self, df):
        """Call a pandas object's ``agg`` method with the cached (kw)args.

        Parameters
        ----------
        df: Series, DataFrame, Rolling or their GroupBy companions
            The pandas object to aggregate.

        Returns
        -------
        scalar, Series, or DataFrame
            The aggregation of the pandas object.

        """
        return df.agg(self.func, *self.args, **self.kwargs)


class RollingWindow(ArgRepr):
    """Simple partial of for calling a pandas object's ``rolling`` method.

    Parameters
    ----------
    *args
        Arguments to pass on to the ``rolling`` method call.
    **kwargs
        Keyword arguments to pass on to the ``rolling`` method call.

    Notes
    -----
    See the pandas `rolling docs <https://pandas.pydata.org/docs/dev/reference
    /api/pandas.core.groupby.DataFrameGroupBy.rolling.html>`_ for a full list
    of (keyword) arguments and an extensive description of usage.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    @overload
    def __call__(self, df: Series) -> Rolling | Window:
        ...

    @overload
    def __call__(self, df: DataFrame) -> Rolling | Window:
        ...

    @overload
    def __call__(self, df: SeriesGroupBy) -> RollingGroupby:
        ...

    @overload
    def __call__(self, df: DataFrameGroupBy) -> RollingGroupby:
        ...

    def __call__(self, df):
        """Call a pandas object`s ``rolling`` method with the cached (kw)args.

        Parameters
        ----------
        df: Series, DataFrame, or their GroupBy companions
            The pandas object to call ``rolling`` on.

        Returns
        -------
        Window, Rolling, or RollingGroupBy
            Depending on the input type.

        """
        return df.rolling(*self.args, **self.kwargs)

from typing import Any, overload
from collections.abc import Callable, Hashable, Mapping
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy, Grouper
from ..misc import ArgRepr
from .types import Labels

type GroupKey = (
    str
    | Callable[[Hashable], Hashable]
    | Grouper
    | Mapping[Hashable, Hashable]
    | ndarray[tuple[int], np.dtype[Any]]
    | Series
)
type GroupKeys = list[GroupKey]


class GroupBy(ArgRepr):
    """Simple partial of a pandas dataframe or series ``groupby`` method.

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
            level: Labels | None = None,
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
        """Call a dataframe or series ``groupby`` method.

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
            self.level,
            as_index=self.as_index,
            sort=self.sort,
            group_keys=self.group_keys,
            observed=self.observed,
            dropna=self.dropna
        )

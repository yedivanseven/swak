from typing import Any
from collections.abc import Hashable, Callable, Mapping
from functools import singledispatchmethod
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Labels, Axis, Errors

type Renamer = Mapping[Hashable, Hashable] | Callable[[Hashable], Hashable]


class Rename(ArgRepr):
    """Simple partial of a pandas dataframe or series ``rename`` method.

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
            mapper: Renamer | Labels | None = None,
            index: Renamer | Labels | None = None,
            columns: Renamer | Labels | None = None,
            axis: Axis = 1,
            level: Hashable | None = None,
            errors: Errors = 'ignore'
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


    @singledispatchmethod
    def __call__(self, df):
        """Rename a pandas dataframe's or series' columns or rows.

        Parameters
        ----------
        df: DataFrame or Series
            The dataframe or series to rename columns or rows of.

        Returns
        -------
        DataFrame or Series
            The dataframe or series with renamed columns or rows.

        Raises
        ------
        TypeError
            When called on an unsuitable object type.

        """
        cls = type(df).__name__
        tmp = 'Cannot rename an object of type {}!'
        msg = tmp.format(cls)
        raise TypeError(msg)

    @__call__.register
    def _(self, df: DataFrame) -> DataFrame:
        return df.rename(
            self.mapper,
            **self.resolved,
            inplace=False,
            level=self.level,
            errors=self.errors
        )

    @__call__.register
    def _(self, df: Series) -> Series:
        return df.rename(
            self.index,
            inplace=False,
            level=self.level,
            errors=self.errors
        )

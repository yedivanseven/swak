from typing import Literal, Any
from collections.abc import Hashable, Callable, Mapping
from pandas import DataFrame
from ..misc import ArgRepr

type Renamer = Mapping[Hashable, Hashable] | Callable[[Hashable], Hashable]


# ToDo: Make this work also for a Series! Maybe with single dispatch?
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

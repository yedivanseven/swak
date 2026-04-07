from typing import Literal, overload
from collections.abc import Hashable, Sequence
from pandas import DataFrame, Series
from ..misc import ArgRepr


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

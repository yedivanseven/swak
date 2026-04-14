from typing import Literal, Any, overload
from collections.abc import Hashable
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Axis, Labels


class DropNA(ArgRepr):
    """A simple partial of a pandas dataframe or series' ``dropna`` method.

    Parameters
    ----------
    axis: 0 or "index", 1 or "columns"
        Determine if rows or columns which contain missing values are removed.
        Defaults to 0.
    how: "any" or "all", optional
        Determine if row or column is removed from DataFrame, when we have at
        least one NA or all NA. Defaults to ``None``.
    thresh: int, optional
        Require that many non-NA values. Cannot be combined with how.
        Defaults to ``None``.
    subset: hashable or sequence, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include. Defaults to ``None``.
    ignore_index: bool, optional
        Defaults to ``True``, thus relabeling the resulting axis as
        0, 1, …, n - 1.

    """

    def __init__(
            self,
            axis: Axis = 0,
            how: Literal['any', 'all'] | None = None,
            thresh: int | None = None,
            subset: Labels | None = None,
            ignore_index: bool = True
    ) -> None:
        self.axis = axis
        self.how = how
        self.thresh = thresh
        self.subset = self.__valid(subset) or None
        self.ignore_index = ignore_index
        super().__init__(
            axis=self.axis,
            how=self.how,
            thresh=self.thresh,
            subset=self.subset,
            ignore_index=self.ignore_index
        )

    @property
    def _how_thresh(self) -> dict[str, Any]:
        if self.how is not None:
            return {'how': self.how}
        if self.thresh is not None:
            return {'thresh': self.thresh}
        return {}


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
        match df:
            case DataFrame():
                return df.dropna(
                    axis=self.axis,
                    **self._how_thresh,
                    subset=self.subset,
                    inplace=False,
                    ignore_index=self.ignore_index
                )
            case Series():
                return df.dropna(
                    axis=0,
                    inplace=False,
                    ignore_index=self.ignore_index
                )
            case _:
                cls = type(df).__name__
                tmp = 'Cannot drop rows or columns from an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

    @staticmethod
    def __valid(labels: Labels) -> list[Hashable]:
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

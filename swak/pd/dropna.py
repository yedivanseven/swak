from typing import Literal, overload
from collections.abc import Hashable,Sequence
from pandas import DataFrame, Series
from ..misc import ArgRepr


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
            # ToDo: Check behavior for explicit thresh=None versus no thresh.
            **({'how': self.how} if self.how else {'thresh': self.thresh}),
            subset=self.subset,
            inplace=False,
            ignore_index=self.ignore_index
        )

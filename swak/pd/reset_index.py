from collections.abc import Hashable, Sequence
from pandas import DataFrame
from ..misc import ArgRepr


# ToDo: ake this work also for Series (maybe with single dispatch)?
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

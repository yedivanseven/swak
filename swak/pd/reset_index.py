from typing import overload
from collections.abc import Hashable
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Labels


class ResetIndex(ArgRepr):
    """Simple partial of a pandas dataframe or series ``reset_index`` method.

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
            level: Labels | None = None,
            drop: bool = False,
            col_level: Hashable = 0,
            col_fill: Hashable = '',
            allow_duplicates: bool = False,
            names: Labels | None = None,
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

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Reset the index of a pandas dataframe or series.

        Parameters
        ----------
        df: DataFrame or Series
            The dataframe or series to reset the index of.

        Returns
        -------
        DataFrame or series
            The dataframe with its index reset.

        Raises
        ------
        TypeError
            When called with an unsuitable object type.

        """
        match df:
            case DataFrame():
                return df.reset_index(
                    self.level,
                    drop=self.drop,
                    inplace=False,
                    col_level=self.col_level,
                    col_fill=self.col_fill,
                    allow_duplicates=self.allow_duplicates,
                    names=self.names
                )
            case Series():
                return df.reset_index(
                    self.level,
                    **({'name': self.names} if self.names is not None else {}),
                    drop=self.drop,
                    inplace=False,
                    allow_duplicates=self.allow_duplicates
                )
            case _:
                cls = type(df).__name__
                tmp = 'Cannot reset the index of an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

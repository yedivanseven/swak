from typing import Any
from collections.abc import Hashable, Iterator
import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series, Index
from ..misc import ArgRepr

type Key = (
    Hashable
    | ndarray[tuple[int], np.dtype[Any]]
    | Series
    | Index
    | Iterator[Hashable]
)
type Keys = list[Key]


# ToDo: Make key, keys work like col, *cols?
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

    """

    def __init__(
            self,
            keys: Key | Keys,
            drop: bool = True,
            append: bool = False,
    ) -> None:
        self.keys = keys
        self.drop = drop
        self.append = append
        super().__init__(keys, drop=drop, append=append)

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
            inplace=False
        )

from collections.abc import Hashable
from pandas import DataFrame
from ..misc import ArgRepr
from .types import Labels


class SetIndex(ArgRepr):
    """Simple partial of a pandas dataframe's ``set_index`` method.

    Parameters
    ----------
    key: hashable or array-like
        This parameter can be either a single column key, a single array of
        the same length as the calling DataFrame, or a list containing an
        arbitrary combination of column keys and arrays.
    *keys: hashable
        Additional columns to include into the index.
    drop : bool, optional
        Delete columns to be used as the new index. Defaults to ``True``.
    append : bool, optional
        Whether to append columns to existing index. Defaults to ``False``

    """

    def __init__(
            self,
            key: Labels,
            *keys: Hashable,
            drop: bool = True,
            append: bool = False,
    ) -> None:
        self.keys = self.__valid(key) + self.__valid(keys)
        self.drop = drop
        self.append = append
        super().__init__(self.keys, drop=drop, append=append)

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

    @staticmethod
    def __valid(cols: Labels) -> list[Hashable]:
        """Ensure that the columns are indeed a sequence of hashables."""
        if isinstance(cols, str):
            return [cols]
        try:
            _ = [hash(col) for col in cols]
        except TypeError:
            _ = hash(cols)
            return [cols]
        return list(cols)

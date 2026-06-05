from typing import overload
from collections.abc import Hashable
from pandas import DataFrame, Series
from ..misc import ArgRepr
from .types import Labels, Keep


class DropDuplicates(ArgRepr):
    """Partial of a pandas dataframe or series' ``drop_duplicates`` method.

    Parameters
    ----------
    subset: hashable or sequence, optional
        Only consider certain columns for identifying duplicates.
        Defaults to ``None``, which considers all columns.
    keep: str or bool, optional
        Determines which duplicates (if any) to keep. Must be one of "first",
        "last" or False. Defaults to "first".
    ignore_index: bool, optional
        If ``True``, the resulting axis will be labeled 0, 1, ... n -1.
        Defaults to ``False``.

    """

    def __init__(
            self,
            subset: Labels | None = None,
            *subsets: Hashable,
            keep: Keep = 'first',
            ignore_index: bool = False,
    ) -> None:
        self.subset = (self.__valid(subset) + self.__valid(subsets)) or None
        self.keep = keep
        self.ignore_index = ignore_index
        super().__init__(
            self.subset,
            keep=keep,
            ignore_index=ignore_index
        )

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Drop duplicates from a pandas series or dataframe.

        Parameters
        ----------
        df: Series or DataFrame
            The object to drop duplicates from.

        Returns
        -------
        Series or DataFrame
            The object with duplicates dropped.

        """
        match df:
            case DataFrame():
                return df.drop_duplicates(
                    self.subset,
                    keep=self.keep,
                    ignore_index=self.ignore_index
                )
            case Series():
                return df.drop_duplicates(
                    keep=self.keep,
                    ignore_index=self.ignore_index
                )
            case _:
                cls = type(df).__name__
                tmp = 'Cannot drop duplicates in an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

    @staticmethod
    def __valid(labels: Labels) -> list[Hashable]:
        """Ensure that the subset(s) are indeed a sequence of hashables."""
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

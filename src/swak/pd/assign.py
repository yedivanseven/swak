from typing import Any
from collections.abc import Mapping, Hashable
from pandas import DataFrame
from ..misc import ArgRepr

type Col = Mapping[Hashable, Any]


class Assign(ArgRepr):
    """Light wrapper around a pandas dataframe's ``assign`` method.

    Parameters
    ----------
    col: dict
        A dictionary with the names of newly created (or overwritten) columns
        as keys. If the values are callable, they are computed on the entire
        dataframe and assigned to the new columns. The callable must not
        the change input dataframe (though pandas doesn’t check it). If the
        values are not callable, e.g., a series, scalar, or array, they are
        simply assigned.
    **cols
        As in the original, the keyword arguments themselves serve as the
        name(s) of the new (or overwritten) column(s) and their values are set
        in the same way.

    """

    def __init__(self, col: Col | None = None, **cols: Any) -> None:
        self.cols = cols if col is None else col | cols
        super().__init__(**self.cols)

    def __call__(self, df: DataFrame) -> DataFrame:
        """Add new columns to a dataframe by calling its ``assign`` method.

        Parameters
        ----------
        df: DataFrame
            The dataframe to add new columns to.

        Returns
        -------
        DataFrame
            The input dataframe with new columns added.

        """
        return df.assign(**self.cols)

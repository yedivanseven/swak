from ..misc import ArgRepr
from polars import DataFrame


class Tail(ArgRepr):
    """Partial pf the polars dataframe ``tail`` method.

    Parameters
    ----------
    n: int, optional
        Number of rows to return, by default 5.

    """

    def __init__(self, n: int = 5) -> None:
        super().__init__(n)
        self.n = n

    def __call__(self, df: DataFrame) -> DataFrame:
        """Return the last `n` rows of a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The polars dataframe to return the tail of.

        Returns
        -------
        DataFrame
            The last `n` rows of `df`.

        """
        return df.tail(self.n)

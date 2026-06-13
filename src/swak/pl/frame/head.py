from ..misc import ArgRepr
from polars import DataFrame


class Head(ArgRepr):
    """Partial pf the polars dataframe ``head`` method.

    Parameters
    ----------
    n: int, optional
        Number of rows to return, by default 5.

    """

    def __init__(self, n: int = 5) -> None:
        super().__init__(n)
        self.n = n

    def __call__(self, df: DataFrame) -> DataFrame:
        """Return the first `n` rows of a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The polars dataframe to return the head of.

        Returns
        -------
        DataFrame
            The first `n` rows of `df`.

        """
        return df.head(self.n)

from collections.abc import Collection
from polars import DataFrame
from polars._typing import IntoExpr, UniqueKeepStrategy
from ...misc import ArgRepr


class Unique(ArgRepr):
    """Partial of the polars dataframe `unique <unique_>`__ method.

    Parameters
    ----------
    subset: IntoExpr
        Column(s) to sort by. Accepts expression input, including selectors.
        Strings are parsed as column names.
    keep: str, optional
        Which of the duplicate rows to keep. must be one of "first", "last",
        "any", or "none". Defaults to "any".
    maintain_order: bool, optional
        Keep the same order as the original DataFrame. This is more expensive
        to compute. Settings this to ``True`` blocks the possibility to run on
        the streaming engine. Defaults to ``False``.


    .. _unique: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.unique.html

    """

    def __init__(
            self,
            subset: IntoExpr | Collection[IntoExpr] | None = None,
            *,
            keep: UniqueKeepStrategy = "any",
            maintain_order: bool = False,
    ) -> None:
        super().__init__(
            subset,
            keep=keep,
            maintain_order=maintain_order
        )
        self.subset = subset
        self.keep = keep
        self.maintain_order = maintain_order

    def __call__(self, df: DataFrame) -> DataFrame:
        """Drop duplicate rows from a polars dataframe.

        Parameters
        ----------
        df: DataFrame
            The dataframe to drop duplicate rows from.

        Returns
        -------
        DataFrame
            The dataframe with duplicate rows dropped.

        """
        return df.unique(
            self.subset,
            keep=self.keep,
            maintain_order=self.maintain_order
        )

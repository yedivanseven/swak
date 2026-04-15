from collections.abc import Iterable, Sequence
from polars import DataFrame
from polars._typing import IntoExpr
from ...misc import ArgRepr


class Sort(ArgRepr):
    """Partial of the polars dataframe `sort <sort_>`__ method.

    Parameters
    ----------
    by: IntoExpr
        Column(s) to sort by. Accepts expression input, including selectors.
        Strings are parsed as column names.
    *more_by: IntoExpr
        Additional columns to sort by, specified as positional arguments.
    descending: bool, optional
        Sort in descending order. When sorting by multiple columns, can be
        specified per column by passing a sequence of booleans.
        Defaults to ``False``.
    nulls_last: bool, optional
        Place null values last. Can be a single boolean applying to all
        columns or a sequence of booleans for per-column control.
        Defaults to ``False``
    multithreaded: bool, optional
        Sort using multiple threads. Defaults to ``True``.
    maintain_order: bool, optional
        Whether the order should be maintained if elements are equal.
        Defaults to ``False``.


    .. _sort: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.sort.html

    """

    def __init__(
            self,
            by: IntoExpr | Iterable[IntoExpr],
            *more_by: IntoExpr,
            descending: bool | Sequence[bool] = False,
            nulls_last: bool | Sequence[bool] = False,
            multithreaded: bool = True,
            maintain_order: bool = False,
    ) -> None:
        super().__init__(
            by,
            *more_by,
            descending=descending,
            nulls_last=nulls_last,
            multithreaded=multithreaded,
            maintain_order=maintain_order
        )
        self.by = by
        self.more_by = more_by
        self.descending = descending
        self.nulls_last = nulls_last
        self.multithreaded = multithreaded
        self.maintain_order = maintain_order

    def __call__(self, df: DataFrame) -> DataFrame:
        """Sort polars dataframe by column values.

        Parameters
        ----------
        df: DataFrame
            The dataframe to sort.

        Returns
        -------
        DataFrame
            The sorted dataframe.

        """
        return df.sort(
            self.by,
            *self.more_by,
            descending=self.descending,
            nulls_last=self.nulls_last,
            multithreaded=self.multithreaded,
            maintain_order=self.maintain_order
        )

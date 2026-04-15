from typing import Any
from collections.abc import Iterable
from numpy import ndarray
from polars import DataFrame, Series
from polars._typing import IntoExprColumn
from ...misc import ArgRepr

type Predicate = (
    bool
    | list[bool]
    | ndarray[bool]
    | Series
    | IntoExprColumn
    | Iterable[IntoExprColumn]
)


class Filter(ArgRepr):
    """Partial of the polars dataframe `filter <filter_>`__ method.

    Parameters
    ----------
    *predicates:
        Expression(s) that evaluate to a boolean Series.
    **constraints
        Filter column(s) given named by the keyword argument itself by the
        supplied value. Constraints will be implicitly combined with other
        filters with a logical and.


    .. _filter: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.filter.html

    """

    def __init__(self, *predicates: Predicate, **constraints: Any) -> None:
        super().__init__(*predicates, **constraints)
        self.predicates = predicates
        self.constraints = constraints

    def __call__(self, df: DataFrame) -> DataFrame:
        """Filter dataframe by `predicates` anf value `constraints`.

        Parameters
        ----------
        df: DataFrame
            The dataframe to filter.

        Returns
        -------
        DataFrame
            The filtered dataframe.

        """
        return df.filter(*self.predicates, **self.constraints)

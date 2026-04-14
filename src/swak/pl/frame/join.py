from collections.abc import Sequence
from polars import DataFrame, Expr
from polars._typing import JoinStrategy, JoinValidation, MaintainOrderJoin
from ...misc import ArgRepr


class Join(ArgRepr):
    """Partial of the polars dataframe `join <join_>`__ method.

    Parameters
    ----------
    on: str
        Name(s) of the join columns in both DataFrames. If set, `left_on` and
        `right_on` should be ``None``. Should not be specified if `how` is
        "cross". Defaults to ``None``.
    how: "inner", "left", "right", "full", "semi", "anti", "cross"
        Join strategy.
    left_on: str, optional
        Name(s) of the left join column(s). Defaults to ``None``.
    right_on: str, optional
        Name(s) of the right join column(s). Defaults to ``None``.
    suffix: str, optional
        Suffix to append to columns with a duplicate name.
        Defaults to "_right".
    validate: "m:m", "m:1", "1:m", "1:1"
        Checks if join is of specified type, many-to-many, many-to-one,
        one-to_many, or one-to-one.
    nulls_equal: bool, optional
        Join on null values. By default, null values will never produce
        matches. Defaults to ``False``.
    coalesce: bool, optional
        Coalescing behavior (merging of join columns). Defaults to ``None``,
        which leaves the behaviour join specific.
    maintain_order: "none", "left", "right", "left_right", "right_left"
        Which dataframe row order to preserve, if any. Do not rely on any
        observed ordering without explicitly setting this parameter, as your
        code may break in a future release. Not specifying any ordering can
        improve performance Supported for inner, left, right and full joins.


    .. _join: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
              polars.DataFrame.join.html

    """

    def __init__(
            self,
            on: str | Expr | Sequence[str | Expr] | None = None,
            how: JoinStrategy = 'inner',
            left_on: str | Expr | Sequence[str | Expr] | None = None,
            right_on: str | Expr | Sequence[str | Expr] | None = None,
            suffix: str = '_right',
            validate: JoinValidation = 'm:m',
            nulls_equal: bool = False,
            coalesce: bool | None = None,
            maintain_order: MaintainOrderJoin | None = None,
    ) -> None:
        self.on = on
        self.how = how.strip().lower()
        self.left_on = left_on
        self.right_on = right_on
        self.suffix = suffix.strip()
        self.validate = validate.strip().lower()
        self.nulls_equal = nulls_equal
        self.coalesce = coalesce
        self.maintain_order = maintain_order
        super().__init__(
            on,
            self.how,
            left_on=left_on,
            right_on=right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=nulls_equal,
            coalesce=coalesce,
            maintain_order=maintain_order,
        )

    def __call__(self, left: DataFrame, right: DataFrame) -> DataFrame:
        """Join two polars dataframes.

        Parameters
        ----------
        left: DataFrame
            Left dataframe in the join.
        right: DataFrame
            Right dataframe in the join.

        Returns
        -------
        DataFrame
            The joined dataframes.

        """
        return left.join(
            right,
            self.on,
            self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order
        )

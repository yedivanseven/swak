from polars import DataFrame
from swak.misc import ArgRepr


class VStack(ArgRepr):
    """Partial of the polars dataframe `vstack <vstack_>`__ method.

    Parameters
    ----------
    in_place: bool, optional
       Whether to modify in place. Defaults to ``False``.


    .. _vstack: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.vstack.html#polars.DataFrame.vstack

    """

    def __init__(self, in_place: bool = False):
        self.in_place = bool(in_place)
        super().__init__(in_place=self.in_place)

    def __call__(self, upper: DataFrame, lower: DataFrame) -> DataFrame:
        """Stack to polars dataframes on top of each other.

        Parameters
        ----------
        upper: DataFrame
            The upper dataframe to be appended to.
        lower: DataFrame
            The lower dataframe being appended to `upper`.

        Returns
        -------
        DataFrame
            The `upper` and `lower` dataframes stacked in top of each other.

        """
        return upper.vstack(lower, in_place=self.in_place)

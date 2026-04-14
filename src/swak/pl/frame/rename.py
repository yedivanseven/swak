from collections.abc import Mapping, Callable
from polars import DataFrame
from swak.misc import ArgRepr


class Rename(ArgRepr):
    """Partial of the polars dataframe `rename <rename_>`__ method.

    Parameters
    ----------
    mapping: Mapping or Callable
        Key value pairs that map from old name to new name, or a function that
        takes the old name as input and returns the new name.
    strict: bool, optional
       Validate that all column names exist in the current schema, and throw
       an exception if any do not. Defaults to ``True``.


    .. _rename: https://docs.pola.rs/api/python/stable/reference/dataframe/api/
                polars.DataFrame.rename.html#polars.DataFrame.rename

    """
    def __init__(
            self,
            mapping: Mapping[str, str] | Callable[[str], str],
            *,
            strict: bool = True,
    ) -> None:
        super().__init__(mapping, strict=strict)
        self.mapping = mapping
        self.strict = strict

    def __call__(self, df: DataFrame) -> DataFrame:
        """Rename a polars dataframe's columns.

        Parameters
        ----------
        df: DataFrame
            The dataframe to rename columns of.

        Returns
        -------
        DataFrame
            The dataframe with renamed columns.

        """
        return df.rename(self.mapping, strict=self.strict)

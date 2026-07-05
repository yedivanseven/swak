from polars import DataFrame


def interpolate(df: DataFrame) -> DataFrame:
    """Call the ``interpolate`` method on a polars dataframe.

    Parameters
    ----------
    df: DataFrame
        The polars dataframe to interpolate.

    Returns
    -------
    DataFrame
        The interpolated polars dataframe.

    """
    return df.interpolate()

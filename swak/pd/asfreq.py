from typing import overload, Literal
from collections.abc import Hashable
from pandas import DataFrame, Series, DateOffset
from ..misc import ArgRepr

type Method =  Literal['backfill', 'bfill', 'pad', 'ffill']


class AsFreq(ArgRepr):
    """Light wrapper around a pandas dataframe or series ``asfreq`` method.

    Parameters
    ----------
    freq: DateOffset or str
        Frequency DateOffset or string.
    method: str, optional
        Method to use for filling holes in re-indexed Series (note this
        does not fill NaNs that already were present). Must be one of
        "pad" / "ffill" or "backfill" / "bfill". Defaults to ``None``.
    how: str, optional
        For PeriodIndex only. Must be one of "start" or "end".
        Defaults to ``None``.
    normalize: bool, optional
        Whether to reset output index to midnight. defaults to ``False``
    fill_value: scalar, optional
        Value to use for missing values, applied during upsampling (note
        this does not fill NaNs that already were present).

    """

    def __init__(
            self,
            freq:  DateOffset | str,
            method: Method | None = None,
            how: Literal['start', 'end'] | None = None,
            normalize: bool = False,
            fill_value: Hashable | None = None,
    ) -> None:
        self.freq = freq
        self.method = method
        self.how = how
        self.normalize = normalize
        self.fill_value = fill_value
        super().__init__(freq, method, how, normalize, fill_value)

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Call the ``asfreq`` method of the passed pandas object.

        Parameters
        ----------
        df: Series or DataFrame
            The pandas object to call ``asfreq`` on with the cached
            (keyword) arguments.

        Returns
        -------
        Series or DataFrame
            The same type as called with.

        """
        return df.asfreq(
            self.freq,
            self.method,
            self.how,
            self.normalize,
            self.fill_value
        )

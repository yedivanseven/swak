from typing import Any, overload
from pandas import DataFrame, Series
from pandas.core.resample import Resampler
from .types import Axis, LimitDirection, LimitArea
from ..misc import ArgRepr


class Interpolate(ArgRepr):
    """Simple partial for calling a pandas object's ``interpolate`` method.

    Parameters
    ----------
    method: str, optional
        Interpolation technique to use.
    axis: int or str, optional
        Axis to interpolate along in the case of a dataframe. Must be one
        of 0, "index", 1, or "columns". Ignored for all other pandas objects.
        Defaults to 0.
    limit: int, optional
        Maximum number of consecutive NaNs to fill. Must be greater than 0.
    inplace: bool, optional
        Update the data in place if possible.
    limit_direction: str, optional
       Consecutive NaNs will be filled in this direction. Must be one of
       "forward", "backward", or "both". Defaults to "forward".
    limit_area : {`None`, 'inside', 'outside'}, default None
            If limit is specified, consecutive NaNs will be filled with this
            restriction.

            * ``None``: No fill restriction.
            * "inside": Only fill NaNs surrounded by valid values
              (interpolate).
            * "outside": Only fill NaNs outside valid values (extrapolate).

    **kwargs : optional
        Keyword arguments to pass on to the interpolating function.

    Note
    ----
    For a full list of (keyword) arguments and their description, see the
    pandas `interpolate documentation <https://pandas.pydata.org/pandas-docs/
    stable/reference/api/pandas.DataFrame.interpolate.html>`_.

    """

    def __init__(
            self,
            method: str = 'linear',
            axis: Axis = 0,
            limit: int | None = None,
            limit_direction: LimitDirection | None = None,
            limit_area: LimitArea | None = None,
            **kwargs: Any
    ) -> None:
        self.method = method
        self.axis = axis
        self.limit = limit
        self.limit_direction = limit_direction
        self.limit_area = limit_area
        self.kwargs = kwargs
        super().__init__(
            method,
            axis,
            limit,
            limit_direction,
            limit_area,
            **kwargs
        )

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: Resampler) -> DataFrame | Series:
        ...

    def __call__(self, df):
        """Call the ``interpolate`` method of a pandas object..

        Parameters
        ----------
        df: Series, DataFrame, or Resampler
            The pandas object to interpolate.

        Returns
        -------
        Series or DataFrame
            Depending on the input type.

        Raises
        ------
        TypeError
            When called on an object that cannot be interpolated.

        """
        match df:
            case DataFrame():
                return df.interpolate(
                    self.method,
                    axis=self.axis,
                    limit=self.limit,
                    inplace=False,
                    limit_direction=self.limit_direction,
                    limit_area=self.limit_area,
                    **self.kwargs
                )
            case Series():
                return df.interpolate(
                    self.method,
                    axis=0,
                    limit=self.limit,
                    inplace=False,
                    limit_direction=self.limit_direction,
                    limit_area=self.limit_area,
                    **self.kwargs
                )
            case Resampler():
                return df.interpolate(
                    self.method,
                    axis=self.axis,
                    limit=self.limit,
                    inplace=False,
                    limit_direction=self.limit_direction,
                    limit_area=self.limit_area,
                    **self.kwargs
                )
            case _:
                cls = type(df).__name__
                tmp = 'Cannot interpolate an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

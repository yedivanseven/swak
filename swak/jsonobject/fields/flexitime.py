from typing import Any, Self
import datetime as dt
import pandas as pd
from pandas import Timestamp, Timedelta

Time = str | dt.date | dt.datetime | Timestamp
Delta = dt.timedelta | Timedelta


# ToDo: Add as_np and try to convert from numpy datetime64
# ToDo: Add polars support
# Todo: Add as_date
class FlexiTime:
    """Flexible wrapper around python's own ``datetime.datetime`` object.

    Parameters
    ----------
    time
        Can be an ISO string of a date or datetime, a ``datetime.date`` or
        a ``datetime.datetime`` object, or a ``pandas.Timestamp``.

    """

    def __init__(self, time: Time | Self) -> None:
        self.as_datetime = dt.datetime.fromisoformat(str(time))

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.as_datetime, attribute)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{str(self)}')"

    def __str__(self) -> str:
        return str(self.as_datetime)

    def __lt__(self, other: Time) -> bool:
        return self.as_datetime < self.__class__(other).as_datetime

    def __le__(self, other: Time) -> bool:
        return self.as_datetime <= self.__class__(other).as_datetime

    def __gt__(self, other: Time) -> bool:
        return self.as_datetime > self.__class__(other).as_datetime

    def __ge__(self, other: Time) -> bool:
        return self.as_datetime >= self.__class__(other).as_datetime

    def __eq__(self, other: Time) -> bool:
        return self.__class__(other).as_datetime == self.as_datetime

    def __ne__(self, other: Time) -> bool:
        return self.__class__(other).as_datetime != self.as_datetime

    def __add__(self, other: Delta) -> Self:
        return self.__class__(self.as_datetime + other)

    def __radd__(self, other: Delta) -> Self:
        return self.__class__(other + self.as_datetime)

    def __sub__(self, other: Delta) -> Self:
        return self.__class__(self.as_datetime - other)

    def __hash__(self) -> int:
        return self.as_datetime.__hash__()

    @property
    def as_json(self) -> str:
        """Representation to appear in a JSON."""
        return str(self)

    @property
    def as_dtype(self) -> Timestamp:
        """Representation in a pandas DataFrame."""
        return pd.to_datetime(self.as_datetime)

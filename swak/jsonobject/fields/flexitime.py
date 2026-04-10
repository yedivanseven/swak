from typing import Any, Self
import datetime as dt
import numpy as np
import pandas as pd
from pandas import Timestamp, Timedelta
from .custom import CustomField

Time = str | dt.date | dt.datetime | Timestamp | np.datetime64
Delta = dt.timedelta | Timedelta | np.timedelta64


# ToDo: Add polars support
class FlexiTime(CustomField):
    """Flexible wrapper around python's own ``datetime.datetime`` object.

    Parameters
    ----------
    time
        Can be an ISO string of a date or datetime, a ``datetime.date``,
        a ``datetime.datetime`` object, a ``np.datetime64`` object,
        a ``pandas.Timestamp``, or a :class:`FlexiDate` object.

    """

    def __init__(self, time: Time | Self) -> None:
        self.as_datetime = dt.datetime.fromisoformat(str(time))

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.as_datetime, attribute)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"

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
    def as_np(self) -> np.datetime64:
        """Representation a numpy datetime object."""
        return np.datetime64(self.as_datetime)

    @property
    def as_date(self) -> dt.date:
        """Date part as datetime.date object."""
        return self.as_datetime.date()

    @property
    def as_json(self) -> str:
        """Representation to appear in a JSON."""
        return str(self)

    @property
    def as_dtype(self) -> Timestamp:
        """Representation in a pandas DataFrame."""
        return pd.to_datetime(self.as_datetime)

    @property
    def as_polars(self) -> dt.datetime:
        """Representation for polars to ingest."""
        return self.as_datetime

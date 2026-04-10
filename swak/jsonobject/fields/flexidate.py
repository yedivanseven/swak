from typing import Any, Self
import datetime as dt
import numpy as np
import pandas as pd
from pandas import Timestamp, Timedelta
from .custom import CustomField

Date = str | dt.date | dt.datetime | Timestamp | np.datetime64
Delta = dt.timedelta | Timedelta | np.timedelta64


# ToDo: Add polars support
class FlexiDate(CustomField):
    """Flexible wrapper around python's own ``datetime.date`` object.

    Parameters
    ----------
    date
        Can be an ISO string of a date or datetime, a ``datetime.date``,
        a ``datetime.datetime`` object, a ``np.datetime64`` object,
        a ``pandas.Timestamp`` or a :class:`FlexiTime` object.

    """

    def __init__(self, date: Date | Self) -> None:
        self.as_date = dt.datetime.fromisoformat(str(date)).date()

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.as_date, attribute)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self!s}')"

    def __str__(self) -> str:
        return str(self.as_date)

    def __lt__(self, other: Date) -> bool:
        return self.as_date < self.__class__(other).as_date

    def __le__(self, other: Date) -> bool:
        return self.as_date <= self.__class__(other).as_date

    def __gt__(self, other: Date) -> bool:
        return self.as_date > self.__class__(other).as_date

    def __ge__(self, other: Date) -> bool:
        return self.as_date >= self.__class__(other).as_date

    def __eq__(self, other: Date) -> bool:
        return self.__class__(other).as_date == self.as_date

    def __ne__(self, other: Date) -> bool:
        return self.__class__(other).as_date != self.as_date

    def __add__(self, other: Delta) -> Self:
        return self.__class__(self.as_date + other)

    def __radd__(self, other: Delta) -> Self:
        return self.__class__(other + self.as_date)

    def __sub__(self, other: Delta) -> Self:
        return self.__class__(self.as_date - other)

    def __hash__(self) -> int:
        return self.as_date.__hash__()

    @property
    def as_datetime(self) -> dt.datetime:
        """Date as a datetime object."""
        return dt.datetime(
            self.as_date.year,
            self.as_date.month,
            self.as_date.day
        )

    @property
    def as_np(self) -> np.datetime64:
        """Representation a numpy datetime object."""
        return np.datetime64(self.as_date)

    @property
    def as_json(self) -> str:
        """Representation to appear in a JSON."""
        return str(self)

    @property
    def as_dtype(self) -> Timestamp:
        """Representation in a pandas DataFrame."""
        return pd.to_datetime(self.as_date)

    @property
    def as_polars(self) -> dt.date:
        """Representation for polars to ingest."""
        return self.as_date

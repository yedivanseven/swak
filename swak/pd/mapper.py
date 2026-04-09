from typing import Literal, Any, overload
from collections.abc import Hashable, Callable, Mapping
from pandas import DataFrame, Series
from ..misc import ArgRepr

type Transform = Mapping[Hashable, Any] | Series | Callable[[Any], Any]


class Mapper(ArgRepr):
    """Partial of a pandas dataframe or series ``map`` method.

    Parameters
    ----------
    func: callable, Mapping, or Series
        Function or mapping in the form of a dictionary or a pandas series.
    na_action: str, optional
        Can take the value "ignore" or ``None``, defaulting to the latter.
    engine: decorator, optional
        The engine to use for a pandas series, Defaults to ``None``.
    **kwargs: Any
        Keyword arguments are pass on to `func`.


    .. _map: https://pandas.pydata.org/pandas-docs/stable/reference/api
             pandas.Series.map.html

    """

    def __init__(
            self,
            func: Transform,
            na_action: Literal['ignore'] | None = None,
            engine: Callable | None = None,
            **kwargs: Any
    ) -> None:
        self.func = func
        self.na_action = na_action
        self.engine = engine
        self.kwargs = kwargs
        name = func if callable(func) else type(func)
        super().__init__(name, na_action, engine, **kwargs)

    @overload
    def __call__(self, df: Series) -> Series:
        ...

    @overload
    def __call__(self, df: DataFrame) -> DataFrame:
        ...

    def __call__(self, df):
        """Called the ``map`` method of a pandas series or dataframe.

        Cached keyword arguments are forwarded to the method call.

        Parameters
        ----------
        df: DataFrame or Series
            Pandas dataframe with the column to call the ``map`` method on.

        Returns
        -------
        DataFrame or Series
            Pandas object with the result or the map operation.

        Raises
        ------
        TypeError
            When called on an unsuitable object type.

        """
        match df:
            case DataFrame():
                return df.map(self.func, self.na_action, **self.kwargs)
            case Series():
                return df.map(
                    self.func,
                    self.na_action,
                    self.engine,
                    **self.kwargs
                )
            case _:
                cls = type(df).__name__
                tmp = 'Cannot map an object of type {}!'
                msg = tmp.format(cls)
                raise TypeError(msg)

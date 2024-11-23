from typing import Any
from pathlib import Path
from pandas import DataFrame
from ..misc import ArgRepr


class ParquetWriter(ArgRepr):
    """Partial of the pandas dataframe ``to_parquet`` method.

    Parameters
    ----------
    path: str, optional
        Path (including file name) to save the parquet file to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    create: bool, optional
        What to do if the directory where the parquet file should be saved does
        not exist. Defaults to ``False``.
    **kwargs
        Keyword arguments passed on to the ``to_parquet`` method call.

    """

    def __init__(
            self,
            path: str = '',
            create: bool = False,
            **kwargs: Any
    ) -> None:
        self.path = str(path).strip()
        self.create = create
        self.kwargs = kwargs
        super().__init__(self.path, create, **kwargs)

    def __call__(self, df: DataFrame, *parts: str) -> tuple[()]:
        """Write a pandas DataFrame to parquet file.

        Parameters
        ----------
        df: DataFrame
            The dataframe to write.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path.resolve()), **self.kwargs)
        return ()

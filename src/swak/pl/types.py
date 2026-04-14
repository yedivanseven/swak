from collections.abc import Iterable
from polars._typing import IntoExpr

type IntoExprs = IntoExpr | Iterable[IntoExpr]

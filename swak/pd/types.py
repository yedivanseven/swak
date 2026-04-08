from typing import Literal
from collections.abc import Sequence, Hashable

type Axis = Literal[0, 'index', 1, 'columns']
type Errors = Literal['ignore', 'raise']
type Engine = Literal['cython', 'numba', 'python']
type Labels = Hashable | Sequence[Hashable]

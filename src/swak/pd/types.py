from typing import Literal
from collections.abc import Sequence, Hashable

type Axis = Literal['index', 'columns', 'rows'] | int
type Errors = Literal['ignore', 'raise']
type Keep = Literal['first', 'last', False]
type Labels = Hashable | Sequence[Hashable]

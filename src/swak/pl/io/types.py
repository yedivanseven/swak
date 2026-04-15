from typing import Literal
from enum import StrEnum

type LiteralLazyStorage = Literal['file', 's3', 'gs', 'az', 'hf']


class LazyStorage(StrEnum):
    FILE = 'file'
    S3 = 's3'
    GCS = 'gs'
    AZURE = 'az'
    HF = 'hf'

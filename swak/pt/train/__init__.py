"""Flexibly parameterized and feature-rich training loop for models.

Allows for warm-up periods as well as early stopping with recovery of the best
checkpoint from CPU memory or disk. Use the provided callback examples to
monitor training progress or derive your own from the respective base classes.
Custom templates for train and test data to seamlessly work with the training
loop are also provided.

"""

from .data import TestDataBase, TrainDataBase
from .trainer import Trainer
from .checkpoints import Checkpoint, InMemory, OnDisk, State
from .callbacks import (
    EpochPrinter,
    TrainPrinter,
    EpochCallback,
    History
)

__all__ = [
    'Trainer',
    'InMemory',
    'OnDisk',
    'EpochPrinter',
    'TrainPrinter',
    'TrainDataBase',
    'TestDataBase',
    'Checkpoint',
    'EpochCallback',
    'State',
    'History'
]

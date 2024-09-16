from typing import TypedDict
from collections.abc import Callable


class History(TypedDict):
    train_loss: list[float]
    test_loss: list[float | None]
    lr: list[float]

type EpochCb = Callable[[int, float, float, list[float]], None]
type TrainCb = Callable[[int, int, float, bool, History], None]

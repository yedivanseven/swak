from abc import ABC, abstractmethod
from ..types import Batch, Batches


class TestDataBase(ABC):

    @property
    @abstractmethod
    def n(self) -> int:
        """The total number of data points."""
        ...

    @abstractmethod
    def sample(self, size: int) -> Batch:
        """Return exactly the same (sub-)sample of the data with given size."""
        ...


class TrainDataBase(TestDataBase):

    @abstractmethod
    def __call__(self, batch_size: int) -> Batches:
        """Return and iterator over tuples of features and target."""
        ...

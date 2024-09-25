from abc import ABC, abstractmethod
from ..types import Batch, Batches

__all__ = [
    'TestDataBase',
    'TrainDataBase'
]


class TestDataBase(ABC):

    @property
    @abstractmethod
    def n(self) -> int:
        """The total number of data points."""
        ...

    @abstractmethod
    def sample(self, size: int) -> Batch:
        """Return a reproducible (sub-)sample of your data with the given size.

        Needed to consistently evaluate (and report) the train and/or test
        error after every epoch of training.

        Parameters
        ----------
        size: int
            Number of data points in (or `batch_size` of) the reproducible
            sample of your data. Every time `step` is called with the same
            `size`, the exact same output should be returned so that train and
            test errors can be used to track training convergence.

        Returns
        -------
        tuple
            A tuple of input tensors to call your model with.
        Tensor
            The matching target values. Must have the same dimensions and sizes
            as the output of your model.

        Important
        ---------
        Even if your model only takes a single tensor as input, this method
        must return a tuple to cover the general case of your model taking
        more than one tensor as input. If this is not the case, simple return
        a 1-tuple of tensors.

        """
        ...


class TrainDataBase(TestDataBase):

    @abstractmethod
    def __call__(self, batch_size: int) -> Batches:
        """Return an iterator over the mini-batches your model is trained on.

        Parameters
        ----------
        batch_size: int
            The (maximum) number of data points in one batch.

        Returns
        -------
        Iterator
            One element yielded by the iterator is a 2-tuple. The first element
            is again a tuple, containing the input tensor(s) to call your model
            with. The second element of the tuple is a single tensor with
            the matching target values. It must have the same dimensions and
            sizes as the output of your model.

        Important
        ---------
        Even if your model only takes a single tensor as input, the first
        element of one tuple yielded by the returned iterator must always be a
        tuple of tensors to cover the general case of your model taking more
        than one tensor as input. If this is not the case, simple make that a
        1-tuple of tensors!

        """
        ...

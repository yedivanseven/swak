import math
from abc import ABC, abstractmethod
from ..types import Batches

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
    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        """Interator over batches from a reproducible sample of your data.

        Needed to consistently evaluate (and report) the train and/or test
        error after every epoch of training. Every time `sample` is called with
        the same arguments, the exact same output should be returned so that
        train and test errors can be used to track training convergence.

        Parameters
        ----------
        batch_size: int
            Number of data points in (or `batch_size` of) the reproducible
            sample of your data. Every time `sample` is called with the same
            `size`, the exact same output should be returned so that train and
            test errors can be used to track training convergence.
        max_n: int, optional
            Total number of data points to return mini-batches for. Defaults to
            ``None``, resulting in all available data points being used.
            To save some time, however, you might want to not use all available
            training data points just for evaluating your model metrics.

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

    def adjust_n_for(
            self,
            batch_size: int,
            step_freq: int = 1,
            n: int | None = None
    ) -> int:
        """Number of data points reduced to be suitably integer-divisible.

        This is a helper method for users to implement the ``__call__`` method
        in the case of `step_freq` > 1. Taking only the returned number of
        data points guarantees that all batches have the same size and that
        there will be no "left-over" batches at the end of the epoch.

        Parameters
        ----------
        batch_size: int
            The desired number of data points in one batch.
        step_freq: int, optional
            In case this number is > 1, the optimizer will accumulate gradients
            for that many batches before taking a step. All batches should be
            of the same size in this case and there shouldn't be any
            "left-over" batches at the end of each epoch. Defaults to 1.
        n: int, optional
            In rare cases, it might be useful to pass in the number of data
            points to adjust rather than taking the number of data points
            returned by ``self.n``, which is the default.

        Returns
        -------
        int
            Reduced number of data points that is guaranteed to be integer
            divisible by the product of `batch_size` and `step_freq`.

        """
        actual_n = self.n if n is None else n
        if step_freq <= 1:
            return actual_n
        super_batch_size = step_freq * batch_size
        return super_batch_size * (actual_n // super_batch_size)

    def adjust_batches_for(
            self,
            batch_size: int,
            step_freq: int= 1,
            n: int | None = None
    ) -> int:
        """Number of batches reduced to be suitably integer-divisible.

        This is a helper method for users to implement the ``__call__`` method
        in the case of `step_freq` > 1. The returned number of batches
        is guaranteed to be integer-divisible by `step_freq` so that no batches
        are "left over" at the end of the epoch.

        Parameters
        ----------
        batch_size: int
            The desired number of data points in one batch.
        step_freq: int, optional
            In case this number is > 1, the optimizer will accumulate gradients
            for that many batches before taking a step. All batches should be
            of the same size in this case and there shouldn't be any
            "left-over" batches at the end of each epoch. Defaults to 1.
        n: int, optional
            In rare cases, it might be useful to pass in the number of data
            points to adjust rather than taking the number of data points
            returned by ``self.n``, which is the default.

        Returns
        -------
        int
            Reduced number of batches that is guaranteed to be integer
            divisible by `step_freq`.

        """
        adjusted_n = self.adjust_n_for(batch_size, step_freq, n)
        return math.ceil(adjusted_n / batch_size)

    @abstractmethod
    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> tuple[int | None, Batches]:
        """Return an iterator over the mini-batches your model is trained on.

        Parameters
        ----------
        batch_size: int
            The (maximum) number of data points in one batch.
        step_freq: int, optional
            In case this number is > 1, the optimizer will accumulate gradients
            for that many batches before taking a step. All batches should be
            of the same size in this case. and there shouldn't be any
            "left-over" batches at the end of each epoch. Defaults to 1.
        epoch: int, optional
            In rare cases, it might be useful to know the current epoch when
            deciding which batches to return. Should start with 1 in the first
            epoch and, therefore, defaults to 0.

        Returns
        -------
        n_batches: int
            Total number of batches the returned iterator will provide.
            If unknown for some reason, also ``None`` can be returned.
        batches: Iterator
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

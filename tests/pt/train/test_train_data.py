import math
import unittest
from swak.pt.types import Batches
from swak.pt.train import TrainDataBase


class TrainData(TrainDataBase):

    def __init__(self, n: int) -> None:
        self.__n = n

    @property
    def n(self) -> int:
        return self.__n

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        pass

    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> tuple[int | None, Batches]:
        pass


class TestAdjustN(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.train = TrainData(self.n)

    def test_step_freq_1_default_n(self):
        for batch_size in range(10):
            n = self.train.adjust_n_for(batch_size)
            self.assertEqual(self.n, n)

    def test_step_freq_1_given_n(self):
        for batch_size in range(10):
            n = self.train.adjust_n_for(batch_size, n=5)
            self.assertEqual(5, n)

    def test_step_freq_2_default_n(self):
        n = self.train.adjust_n_for(1, 2)
        self.assertEqual(6, n)
        n = self.train.adjust_n_for(2, 2)
        self.assertEqual(4, n)
        n = self.train.adjust_n_for(3, 2)
        self.assertEqual(6, n)

    def test_step_freq_3_default_n(self):
        n = self.train.adjust_n_for(1, 3)
        self.assertEqual(6, n)
        n = self.train.adjust_n_for(2, 3)
        self.assertEqual(6, n)
        n = self.train.adjust_n_for(3, 3)
        self.assertEqual(0, n)

    def test_step_freq_2_given_n(self):
        n = self.train.adjust_n_for(1, 2,15)
        self.assertEqual(14, n)
        n = self.train.adjust_n_for(2, 2, 15)
        self.assertEqual(12, n)
        n = self.train.adjust_n_for(3, 2, 15)
        self.assertEqual(12, n)
        n = self.train.adjust_n_for(4, 2, 15)
        self.assertEqual(8, n)

    def test_step_freq_3_given_n(self):
        n = self.train.adjust_n_for(1, 3, 15)
        self.assertEqual(15, n)
        n = self.train.adjust_n_for(2, 3, 15)
        self.assertEqual(12, n)
        n = self.train.adjust_n_for(3, 3, 15)
        self.assertEqual(9, n)
        n = self.train.adjust_n_for(4, 3, 15)
        self.assertEqual(12, n)
        n = self.train.adjust_n_for(5, 3, 15)
        self.assertEqual(15, n)


class TestAdjustBatches(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.train = TrainData(self.n)

    def test_step_freq_1_default_n(self):
        for batch_size in range(1, 10):
            n_batches = self.train.adjust_batches_for(batch_size)
            self.assertEqual(math.ceil(self.n / batch_size), n_batches)

    def test_step_freq_1_given_n(self):
        for batch_size in range(1, 10):
            n_batches = self.train.adjust_batches_for(batch_size, n=5)
            self.assertEqual(math.ceil(5 / batch_size), n_batches)

    def test_step_freq_2_default_n(self):
        n_batches = self.train.adjust_batches_for(1, 2)
        self.assertEqual(6, n_batches)
        n_batches = self.train.adjust_batches_for(2, 2)
        self.assertEqual(2, n_batches)
        n_batches = self.train.adjust_batches_for(3, 2)
        self.assertEqual(2, n_batches)

    def test_step_freq_3_default_n(self):
        n_batches = self.train.adjust_batches_for(1, 3)
        self.assertEqual(6, n_batches)
        n_batches = self.train.adjust_batches_for(2, 3)
        self.assertEqual(3, n_batches)
        n_batches = self.train.adjust_batches_for(3, 3)
        self.assertEqual(0, n_batches)

    def test_step_freq_2_given_n(self):
        n_batches = self.train.adjust_batches_for(1, 2,15)
        self.assertEqual(14, n_batches)
        n_batches = self.train.adjust_batches_for(2, 2, 15)
        self.assertEqual(6, n_batches)
        n_batches = self.train.adjust_batches_for(3, 2, 15)
        self.assertEqual(4, n_batches)
        n_batches = self.train.adjust_batches_for(4, 2, 15)
        self.assertEqual(2, n_batches)

    def test_step_freq_3_given_n(self):
        n = self.train.adjust_batches_for(1, 3, 15)
        self.assertEqual(15, n)
        n = self.train.adjust_batches_for(2, 3, 15)
        self.assertEqual(6, n)
        n = self.train.adjust_batches_for(3, 3, 15)
        self.assertEqual(3, n)
        n = self.train.adjust_batches_for(4, 3, 15)
        self.assertEqual(3, n)
        n = self.train.adjust_batches_for(5, 3, 15)
        self.assertEqual(3, n)


if __name__ == '__main__':
    unittest.main()

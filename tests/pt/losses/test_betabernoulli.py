import unittest
import torch as pt
from swak.pt.losses import BetaBernoulliLoss, _BaseLoss


class TestBetaBernoulli(unittest.TestCase):

    def setUp(self) -> None:
        self.a = pt.tensor([[3.0], [4.0]])
        self.b = pt.tensor([[5.0], [6.0]])
        self.y = pt.tensor([[1.0], [0.0]])
        self.expected = -pt.tensor([[3/8], [6/10]]).log()
        self.zero = pt.tensor(0.0)

    def test_is_loss(self):
        self.assertIsInstance(BetaBernoulliLoss(), _BaseLoss)

    def test_default(self):
        loss = BetaBernoulliLoss()
        actual = loss(self.a, self.b, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = BetaBernoulliLoss('mean')
        actual = loss(self.a, self.b, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = BetaBernoulliLoss('sum')
        actual = loss(self.a, self.b, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = BetaBernoulliLoss('none')
        actual = loss(self.a, self.b, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_alpha(self):
        loss = BetaBernoulliLoss()
        a = pt.tensor([0.0])
        b = pt.tensor([3.0])
        y = pt.tensor([0.0])
        actual = loss(a, b, y)
        pt.testing.assert_close(actual, self.zero)

    def test_0_beta(self):
        loss = BetaBernoulliLoss()
        a = pt.tensor([3.0])
        b = pt.tensor([0.0])
        y = pt.tensor([1.0])
        actual = loss(a, b, y)
        pt.testing.assert_close(actual, self.zero)

    def test_0_alpha_0_beta_1_target(self):
        loss = BetaBernoulliLoss()
        a = pt.tensor([0.0])
        b = pt.tensor([0.0])
        y = pt.tensor([1.0])
        actual = loss(a, b, y)
        pt.testing.assert_close(actual, self.zero)

    def test_0_alpha_0_beta_0_target(self):
        loss = BetaBernoulliLoss()
        a = pt.tensor([0.0])
        b = pt.tensor([0.0])
        y = pt.tensor([0.0])
        actual = loss(a, b, y)
        pt.testing.assert_close(actual, self.zero)


if __name__ == '__main__':
    unittest.main()

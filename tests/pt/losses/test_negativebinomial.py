import unittest
import torch as pt
from swak.pt.losses import NegativeBinomialLoss, _BaseLoss


class TestNegativeBinomial(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = pt.tensor([[2.0], [1.0]])
        self.sigma = pt.tensor([[4.0], [2.0]]).sqrt()
        self.y = pt.tensor([[1.0], [2.0]])
        self.expected = pt.tensor([[4.0], [8.0]]).log()

    def test_is_loss(self):
        self.assertIsInstance(NegativeBinomialLoss(), _BaseLoss)

    def test_default(self):
        loss = NegativeBinomialLoss()
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = NegativeBinomialLoss('mean')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = NegativeBinomialLoss('sum')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = NegativeBinomialLoss('none')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_mu(self):
        loss = NegativeBinomialLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_sigma_equal_mu(self):
        loss = NegativeBinomialLoss('none')
        actual = loss(self.mu, self.mu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = NegativeBinomialLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, self.sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma(self):
        loss = NegativeBinomialLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target(self):
        loss = NegativeBinomialLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_sigma_equal_mu_0_target(self):
        loss = NegativeBinomialLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, self.mu, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_target(self):
        loss = NegativeBinomialLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()

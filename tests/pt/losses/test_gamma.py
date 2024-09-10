import unittest
import torch as pt
from swak.pt.losses import GammaLoss, _BaseLoss


class TestGamma(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = pt.tensor([[2.0], [3.0]])
        self.sigma = pt.tensor([[2.0], [3.0]])
        self.y = pt.tensor([[2.0], [3.0]])
        self.expected = self.y.log() + 1.0

    def test_is_loss(self):
        self.assertIsInstance(GammaLoss(), _BaseLoss)

    def test_default(self):
        loss = GammaLoss()
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = GammaLoss('mean')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = GammaLoss('sum')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = GammaLoss('none')
        actual = loss(self.mu, self.sigma, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_mu(self):
        loss = GammaLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma(self):
        loss = GammaLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = GammaLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, self.sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma(self):
        loss = GammaLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target(self):
        loss = GammaLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma_0_target(self):
        loss = GammaLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_target(self):
        loss = GammaLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()

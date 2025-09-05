import unittest
import torch as pt
import torch.distributions as ptd
from swak.pt.losses import LogNormalLoss, _BaseLoss


class TestLogNormal(unittest.TestCase):

    def setUp(self) -> None:
        mu = pt.tensor([[2.34], [1.123]])
        sigma = pt.tensor([[4.56], [3.45]]).sqrt()
        self.mean = (mu + 0.5 * sigma.pow(2.0)).exp()
        var = (sigma.pow(2).exp() - 1) * (2 * mu + sigma.pow(2.0)).exp()
        self.y = pt.tensor([[1.23], [2.34]])
        self.stddev = var.sqrt()
        self.expected = -ptd.LogNormal(mu, sigma).log_prob(self.y)

    def test_is_loss(self):
        self.assertIsInstance(LogNormalLoss(), _BaseLoss)

    def test_default(self):
        loss = LogNormalLoss()
        actual = loss(self.mean ,self.stddev, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = LogNormalLoss('mean')
        actual = loss(self.mean, self.stddev, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = LogNormalLoss('sum')
        actual = loss(self.mean, self.stddev, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = LogNormalLoss('none')
        actual = loss(self.mean, self.stddev, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_mu(self):
        loss = LogNormalLoss('none')
        mean = pt.tensor([[0.0], [0.0]])
        actual = loss(mean, self.stddev, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_sigma_equal_mu(self):
        loss = LogNormalLoss('none')
        actual = loss(self.mean, self.mean, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = LogNormalLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mean, self.stddev, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma(self):
        loss = LogNormalLoss('none')
        mean = pt.tensor([[0.0], [0.0]])
        stddev = pt.tensor([[0.0], [0.0]])
        actual = loss(mean, stddev, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target(self):
        loss = LogNormalLoss('none')
        mean = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mean, self.stddev, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_sigma_equal_mu_0_target(self):
        loss = LogNormalLoss('none')
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mean, self.mean, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_target(self):
        loss = LogNormalLoss('none')
        mean = pt.tensor([[0.0], [0.0]])
        stddev = pt.tensor([[0.0], [0.0]])
        y = pt.tensor([[0.0], [0.0]])
        actual = loss(mean, stddev, y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()

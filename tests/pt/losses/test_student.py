import unittest
import math
import torch as pt
from swak.pt.losses import StudentLoss, _BaseLoss


class TestStudent(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = pt.tensor([[3.], [0.]])
        self.sigma = pt.tensor([[0.5], [2.0]])
        self.nu = pt.tensor([[4.], [1.]])
        self.y = pt.tensor([[3.], [2.]])
        self.expected = pt.tensor([[4./3.], [4. * math.pi]]).log()

    def test_is_loss(self):
        self.assertIsInstance(StudentLoss(), _BaseLoss)

    def test_default(self):
        loss = StudentLoss()
        actual = loss(self.mu, self.sigma, self.nu, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = StudentLoss('mean')
        actual = loss(self.mu, self.sigma, self.nu, self.y)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = StudentLoss('sum')
        actual = loss(self.mu, self.sigma, self.nu, self.y)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = StudentLoss('none')
        actual = loss(self.mu, self.sigma, self.nu, self.y)
        pt.testing.assert_close(actual, self.expected)

    def test_0_mu(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, self.nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma(self):
        loss = StudentLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, self.nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_nu(self):
        loss = StudentLoss('none')
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, self.sigma, nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = StudentLoss('none')
        actual = loss(self.mu, self.sigma, self.nu, self.mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, self.nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_nu(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, self.nu, mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma_0_nu(self):
        loss = StudentLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma_0_target(self):
        loss = StudentLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, self.nu, self.mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_nu_0_target(self):
        loss = StudentLoss('none')
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, self.sigma, nu, self.mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_nu(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, nu, self.y)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_target(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, self.nu, mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_nu_0_target(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, self.sigma, nu, mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_sigma_0_nu_0_target(self):
        loss = StudentLoss('none')
        sigma = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(self.mu, sigma, nu, self.mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_sigma_0_nu_0_target(self):
        loss = StudentLoss('none')
        mu = pt.tensor([[0.0], [0.0]])
        sigma = pt.tensor([[0.0], [0.0]])
        nu = pt.tensor([[0.0], [0.0]])
        actual = loss(mu, sigma, nu, mu)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()

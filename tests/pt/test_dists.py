import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.dists import MuSigmaGamma, MuSigmaNegativeBinomial


class TestMuSigmaGamma(unittest.TestCase):

    def test_float(self):
        expected_mu = pt.tensor(5.7)
        expected_sigma = pt.tensor(2.1)
        dist = MuSigmaGamma(5.7, 2.1)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_0d(self):
        expected_mu = pt.tensor(5.7)
        expected_sigma = pt.tensor(2.1)
        dist = MuSigmaGamma(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_1d(self):
        expected_mu = pt.ones(3) * 5.7
        expected_sigma = pt.ones(3) * 2.1
        dist = MuSigmaGamma(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_2d(self):
        expected_mu = pt.ones(3, 4) * 5.7
        expected_sigma = pt.ones(3, 4) * 2.1
        dist = MuSigmaGamma(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_3d(self):
        expected_mu = pt.ones(3, 4, 5) * 5.7
        expected_sigma = pt.ones(3, 4, 5) * 2.1
        dist = MuSigmaGamma(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_4d(self):
        expected_mu = pt.ones(2, 3, 4, 5) * 5.7
        expected_sigma = pt.ones(2, 3, 4, 5) * 2.1
        dist = MuSigmaGamma(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    @patch('torch.distributions.Gamma.__init__')
    def test_args_transform(self, mock):
        mu = 5.7
        sigma = 2.1
        alpha = mu**2 / sigma**2
        beta = mu / sigma**2
        _ = MuSigmaGamma(mu, sigma, True)
        mock.assert_called_once_with(alpha, beta, True)


class TestMuNegativeBinomial(unittest.TestCase):

    def test_float(self):
        expected_mu = pt.tensor(5.7)
        expected_sigma = pt.tensor(3.1)
        dist = MuSigmaNegativeBinomial(5.7, 3.1)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_0d(self):
        expected_mu = pt.tensor(5.7)
        expected_sigma = pt.tensor(3.1)
        dist = MuSigmaNegativeBinomial(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_1d(self):
        expected_mu = pt.ones(3) * 5.7
        expected_sigma = pt.ones(3) * 3.1
        dist = MuSigmaNegativeBinomial(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_2d(self):
        expected_mu = pt.ones(3, 4) * 5.7
        expected_sigma = pt.ones(3, 4) * 3.1
        dist = MuSigmaNegativeBinomial(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_3d(self):
        expected_mu = pt.ones(3, 4, 5) * 5.7
        expected_sigma = pt.ones(3, 4, 5) * 3.1
        dist = MuSigmaNegativeBinomial(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    def test_mean_4d(self):
        expected_mu = pt.ones(2, 3, 4, 5) * 5.7
        expected_sigma = pt.ones(2, 3, 4, 5) * 3.1
        dist = MuSigmaNegativeBinomial(expected_mu, expected_sigma)
        pt.testing.assert_close(dist.mean, expected_mu)
        pt.testing.assert_close(dist.stddev, expected_sigma)

    @patch('torch.distributions.NegativeBinomial.__init__')
    def test_args_transform(self, mock):
        mu = 5.7
        sigma = 3.1
        total_counts = mu**2 / (sigma**2 - mu)
        probs = 1.0 - mu / sigma**2
        _ = MuSigmaNegativeBinomial(mu, sigma, True)
        mock.assert_called_once_with(total_counts, probs, validate_args=True)


if __name__ == '__main__':
    unittest.main()

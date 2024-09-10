import unittest
import torch as pt
from swak.pt.losses import TweedieLoss, _BaseLoss


class TestTweedie(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = pt.tensor([[[4.0, 9.0]]])
        self.p = pt.tensor([[[1.5, 1.5]]])
        self.target = pt.tensor([[[9.0, 4.0]]])
        self.expected = pt.tensor([[[2.0, 4./3.]]])

    def test_is_loss(self):
        self.assertIsInstance(TweedieLoss(), _BaseLoss)

    def test_default(self):
        loss = TweedieLoss()
        actual = loss(self.mu, self.p, self.target)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_mean(self):
        loss = TweedieLoss('mean')
        actual = loss(self.mu, self.p, self.target)
        pt.testing.assert_close(actual, self.expected.mean())

    def test_sum(self):
        loss = TweedieLoss('sum')
        actual = loss(self.mu, self.p, self.target)
        pt.testing.assert_close(actual, self.expected.sum())

    def test_none(self):
        loss = TweedieLoss('none')
        actual = loss(self.mu, self.p, self.target)
        pt.testing.assert_close(actual, self.expected)

    def test_0_mu(self):
        loss = TweedieLoss('none')
        actual = loss(pt.zeros_like(self.mu), self.p, self.target)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target(self):
        loss = TweedieLoss('none')
        actual = loss(self.mu, self.p, pt.zeros_like(self.target))
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target(self):
        loss = TweedieLoss('none')
        actual = loss(
            pt.zeros_like(self.mu),
            self.p,
            pt.zeros_like(self.target)
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_clamp_to_1(self):
        loss = TweedieLoss()
        actual = loss(self.mu, pt.zeros_like(self.p), self.target)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_clamp_to_1(self):
        loss = TweedieLoss('none')
        actual = loss(
            pt.zeros_like(self.mu),
            pt.zeros_like(self.p),
            self.target
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target_clamp_to_1(self):
        loss = TweedieLoss('none')
        actual = loss(
            self.mu,
            pt.zeros_like(self.p),
            pt.zeros_like(self.target)
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target_clamp_to_1(self):
        loss = TweedieLoss('none')
        actual = loss(
            pt.zeros_like(self.mu),
            pt.zeros_like(self.p),
            pt.zeros_like(self.target)
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_clamp_to_2(self):
        loss = TweedieLoss('none')
        actual = loss(self.mu, pt.ones_like(self.p) * 3., self.target)
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_clamp_to_2(self):
        loss = TweedieLoss('none')
        actual = loss(
            pt.zeros_like(self.mu),
            pt.ones_like(self.p) * 3.,
            self.target
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_target_clamp_to_2(self):
        loss = TweedieLoss('none')
        actual = loss(
            self.mu,
            pt.ones_like(self.p) * 3.,
            pt.zeros_like(self.target)
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())

    def test_0_mu_0_target_clamp_to_2(self):
        loss = TweedieLoss('none')
        actual = loss(
            pt.zeros_like(self.mu),
            pt.ones_like(self.p) * 3.,
            pt.zeros_like(self.target)
        )
        self.assertFalse(actual.isinf().any().item())
        self.assertFalse(actual.isnan().any().item())


if __name__ == '__main__':
    unittest.main()

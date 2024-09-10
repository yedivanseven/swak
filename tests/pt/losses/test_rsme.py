import unittest
import torch as pt
from swak.pt.losses import RMSELoss


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.loss = RMSELoss()

    def test_module(self):
        self.assertIsInstance(self.loss, pt.nn.Module)

    def test_has_reduction(self):
        self.assertTrue(hasattr(self.loss, 'reduction'))

    def test_reduction(self):
        self.assertEqual(self.loss.reduction, 'mean')

    def test_has_epsilon(self):
        self.assertTrue(hasattr(self.loss, 'eps'))

    def test_epsilon(self):
        pt.testing.assert_close(self.loss.eps, pt.tensor(1.e-8))

    def test_has_forward(self):
        self.assertTrue(hasattr(self.loss, 'forward'))

    def test_forward_callable(self):
        self.assertTrue(callable(self.loss.forward))

    def test_callable(self):
        self.assertTrue(callable(self.loss))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.loss = RMSELoss('sum', 1.0)

    def test_reduction(self):
        self.assertEqual(self.loss.reduction, 'sum')

    def test_epsilon(self):
        pt.testing.assert_close(self.loss.eps, pt.tensor(1.0))


class TestDimensionality(unittest.TestCase):

    def setUp(self) -> None:
        self.tensor0d = pt.tensor(1).float()
        self.target0d = pt.tensor(3).float()
        self.tensor1d = pt.tensor([4, 5]).float()
        self.target1d = pt.tensor([1, 1]).float()
        self.tensor2d = pt.tensor([[4, 5]]).float()
        self.target2d = pt.tensor([[1, 1]]).float()
        self.tensor3d = pt.tensor([[[4, 5]]]).float()
        self.target3d = pt.tensor([[[1, 1]]]).float()

    def test_raises_on_wrong_reduction(self):
        rmse = RMSELoss('hello')
        with self.assertRaises(ValueError):
            _ = rmse(self.tensor0d, self.target0d)

    def test_0d(self):
        rmse = RMSELoss('none')
        loss = rmse(self.tensor0d, self.target0d)
        self.assertIsInstance(loss, pt.Tensor)
        self.assertEqual(0, loss.dim())
        self.assertEqual(pt.Size([]), loss.shape)

    def test_1d(self):
        rmse = RMSELoss('none')
        loss = rmse(self.tensor1d, self.target1d)
        self.assertEqual(1, loss.dim())
        self.assertEqual(pt.Size([2]), loss.shape)

    def test_2d(self):
        rmse = RMSELoss('none')
        loss = rmse(self.tensor2d, self.target2d)
        self.assertEqual(2, loss.dim())
        self.assertEqual(pt.Size([1, 2]), loss.shape)

    def test_3d(self):
        rmse = RMSELoss('none')
        loss = rmse(self.tensor3d, self.target3d)
        self.assertEqual(3, loss.dim())
        self.assertEqual(pt.Size([1, 1, 2]), loss.shape)


class TestValue(unittest.TestCase):

    def setUp(self) -> None:
        self.tensor = pt.tensor([[[2., 2.]]])
        self.target = pt.tensor([[[1., 1.]]])

    def test_default(self):
        loss = RMSELoss()
        actual = loss(self.tensor, self.target)
        expected = pt.tensor(1.0)
        pt.testing.assert_close(actual, expected)

    def test_mean(self):
        loss = RMSELoss('mean')
        actual = loss(self.tensor, self.target)
        expected = pt.tensor(1.0)
        pt.testing.assert_close(actual, expected)

    def test_sum(self):
        loss = RMSELoss('sum')
        actual = loss(self.tensor, self.target)
        expected = pt.tensor(2.0).sqrt()
        pt.testing.assert_close(actual, expected)

    def test_none(self):
        loss = RMSELoss('none')
        actual = loss(self.tensor, self.target)
        expected = pt.tensor([[[1.0, 1.0]]])
        pt.testing.assert_close(actual, expected)

    def test_zeroes(self):
        loss = RMSELoss()
        actual = loss(self.target, self.target)
        expected = pt.tensor(1e-4)
        pt.testing.assert_close(actual, expected)

    def test_clamping(self):
        loss = RMSELoss(eps=4)
        actual = loss(self.tensor, self.target)
        expected = pt.tensor(2.)
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

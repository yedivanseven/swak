import unittest
from unittest.mock import patch
import torch as pt
from torch.nn import Linear
from swak.pt.embed import LinearEmbedder


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = LinearEmbedder(4)

    def test_has_out_dim(self):
        self.assertTrue(hasattr(self.embed, 'out_dim'))

    def test_out_dim(self):
        self.assertIsInstance(self.embed.out_dim, int)
        self.assertEqual(4, self.embed.out_dim)

    def test_has_inp_dim(self):
        self.assertTrue(hasattr(self.embed, 'inp_dim'))

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(1, self.embed.inp_dim)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.embed, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.embed.kwargs)

    def test_has_embed(self):
        self.assertTrue(hasattr(self.embed, 'embed'))

    def test_embed(self):
        self.assertIsInstance(self.embed.embed, Linear)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = LinearEmbedder(4)
        mock.assert_called_once_with(1, 4)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, LinearEmbedder)
        self.assertEqual(self.embed.out_dim, new.out_dim)
        self.assertEqual(self.embed.inp_dim, new.inp_dim)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = LinearEmbedder(4, 2, bias=False)

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(2, self.embed.inp_dim)

    def test_kwargs(self):
        self.assertDictEqual({'bias': False}, self.embed.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = LinearEmbedder(4, bias=False)
        mock.assert_called_once_with(1, 4, bias=False)

    def test_call_new(self):
        new = self.embed.new(8,4, bias=True)
        self.assertEqual(8, new.out_dim)
        self.assertEqual(4, new.inp_dim)
        self.assertDictEqual({'bias': True}, new.kwargs)


class TestUsageSingleFeature(unittest.TestCase):

    def setUp(self):
        self.embed = LinearEmbedder(4)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(1)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 1)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(2, 3, 1)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([2, 3, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 1)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([1, 2, 3, 4]), actual.shape)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 1)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 0, 4]), actual.shape)

    @patch('torch.nn.Linear.forward')
    def test_linear_called(self, linear):
        inp = pt.ones(1)
        linear.return_value = pt.ones(1, 4)
        embed = LinearEmbedder(4)
        _ = embed(inp)
        linear.assert_called_once_with(inp)


class TestUsageMultiFeature(unittest.TestCase):

    def setUp(self):
        self.embed = LinearEmbedder(4, 2)

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([3, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(2, 3, 2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([2, 3, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([1, 2, 3, 4]), actual.shape)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([3, 0, 4]), actual.shape)

    @patch('torch.nn.Linear.forward')
    def test_linear_called(self, linear):
        inp = pt.ones(2)
        linear.return_value = pt.tensor([[1.0, 1.0, 0.0, 0.0]])
        embed = LinearEmbedder(4, 2)
        _ = embed(inp)
        linear.assert_called_once_with(inp)


if __name__ == '__main__':
    unittest.main()

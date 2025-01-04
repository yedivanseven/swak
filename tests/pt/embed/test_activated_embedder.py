import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.misc import identity
from swak.pt.embed import ActivatedEmbedder


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = ActivatedEmbedder(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.embed, 'activate'))

    def test_activate(self):
        self.assertIs(self.embed.activate, identity)

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
        self.assertIsInstance(self.embed.embed, ptn.Linear)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = ActivatedEmbedder(4)
        mock.assert_called_once_with(1, 4)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called_on_instantiation(self, linear):
        activate = ptn.PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = ActivatedEmbedder(4, activate)
            self.assertEqual(1, mock.call_count)
            self.assertEqual(1, linear.call_count)

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        mock.assert_called_once_with()

    def test_reset_parameters_called_on_activation(self):
        embed = ActivatedEmbedder(4, ptn.PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as activate:
            embed.reset_parameters()
            self.assertEqual(1, activate.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, ActivatedEmbedder)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertIs(new.activate, self.embed.activate)
        self.assertEqual(self.embed.inp_dim, new.inp_dim)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = ActivatedEmbedder(4, ptn.functional.relu, 2, bias=False)

    def test_activate(self):
        self.assertIs(self.embed.activate, ptn.functional.relu)

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(2, self.embed.inp_dim)

    def test_kwargs(self):
        self.assertDictEqual({'bias': False}, self.embed.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = ActivatedEmbedder(4, bias=False)
        mock.assert_called_once_with(1, 4, bias=False)

    def test_call_new(self):
        new = self.embed.new(8, ptn.functional.gelu, 4, bias=True)
        self.assertEqual(8, new.mod_dim)
        self.assertIs(new.activate, ptn.functional.gelu)
        self.assertEqual(4, new.inp_dim)
        self.assertDictEqual({'bias': True}, new.kwargs)


class TestUsageSingleFeature(unittest.TestCase):

    def setUp(self):
        self.embed = ActivatedEmbedder(4, bias=False)
        self.embed.embed.weight.data = pt.ones(4, 1)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(1)
        actual = self.embed(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 1)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward', return_value=pt.ones(1, 4))
    def test_linear_called(self, linear):
        inp = pt.ones(1)
        _ = self.embed(inp)
        linear.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.embed.activate = mock
        inp = pt.ones(1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        pt.testing.assert_close(actual, pt.ones(4))


class TestUsageMultiFeature(unittest.TestCase):

    def setUp(self):
        self.embed = ActivatedEmbedder(4, inp_dim=2, bias=False)
        self.embed.embed.weight.data = pt.ones(4, 2)

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        expected = pt.ones(4) * 2
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4) * 2
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward', return_value=pt.ones(4))
    def test_linear_called(self, linear):
        inp = pt.ones(2)
        _ = self.embed(inp)
        linear.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(4))
        self.embed.activate = mock
        inp = pt.ones(2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        pt.testing.assert_close(actual, pt.ones(4) * 2)


if __name__ == '__main__':
    unittest.main()

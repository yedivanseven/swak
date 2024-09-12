import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Sigmoid, Linear, GELU, ELU, Dropout, AlphaDropout
from swak.pt.misc import identity
from swak.pt.embed import GatedResidualEmbedder


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.embed, 'activate'))

    def test_activate(self):
        self.assertIsInstance(self.embed.activate, ELU)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.embed, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, Sigmoid)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.embed, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.embed.drop, Dropout)

    def test_has_inp_dim(self):
        self.assertTrue(hasattr(self.embed, 'inp_dim'))

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(1, self.embed.inp_dim)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.embed, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.embed.kwargs)

    def test_has_project(self):
        self.assertTrue(hasattr(self.embed, 'project'))

    def test_project(self):
        self.assertIsInstance(self.embed.project, Linear)

    def test_has_widen(self):
        self.assertTrue(hasattr(self.embed, 'widen'))

    def test_widen(self):
        self.assertIsInstance(self.embed.widen, Linear)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedResidualEmbedder(4)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((1, 4), args_1[0])
        self.assertTupleEqual((4, 8), args_2[0])

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, GatedResidualEmbedder)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertIsInstance(self.embed.activate, ELU)
        self.assertIsInstance(self.embed.gate, Sigmoid)
        self.assertIsInstance(self.embed.drop, Dropout)
        self.assertEqual(self.embed.inp_dim, new.inp_dim)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(
            4,
            GELU(),
            ELU(),
            AlphaDropout(),
            2,
            bias=False
        )

    def test_activate(self):
        self.assertIsInstance(self.embed.activate, GELU)

    def test_gate(self):
        self.assertIsInstance(self.embed.gate, ELU)

    def test_drop(self):
        self.assertIsInstance(self.embed.drop, AlphaDropout)

    def test_inp_dim(self):
        self.assertIsInstance(self.embed.inp_dim, int)
        self.assertEqual(2, self.embed.inp_dim)

    def test_kwargs(self):
        self.assertDictEqual({'bias': False}, self.embed.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedResidualEmbedder(4, inp_dim=2, bias=False)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((2, 4), args_1[0])
        self.assertTupleEqual((4, 8), args_2[0])
        self.assertDictEqual({'bias': False}, args_1[1])
        self.assertDictEqual({'bias': False}, args_2[1])

    def test_call_new(self):
        new = self.embed.new(8, GELU(), ELU(), AlphaDropout(), 2, bias=False)
        self.assertIsInstance(new, GatedResidualEmbedder)
        self.assertEqual(8, new.mod_dim)
        self.assertIsInstance(self.embed.activate, GELU)
        self.assertIsInstance(self.embed.gate, ELU)
        self.assertIsInstance(self.embed.drop, AlphaDropout)
        self.assertEqual(2, new.inp_dim)
        self.assertDictEqual({'bias': False}, new.kwargs)


class TestUsageSingleFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(4, identity, identity, bias=False)
        self.embed.project.weight.data = pt.ones(4, 1)
        self.embed.widen.weight.data = pt.ones(8, 4) / 4

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

    def test_project_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.project.forward = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        mock.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.activate = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward', return_value=pt.ones(3, 4))
    def test_drop_called(self, drop):
        project = Mock(return_value=pt.ones(3, 4))
        self.embed.project.forward = project
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = drop.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_expand_called(self):
        mock = Mock(return_value=pt.ones(3, 8))
        self.embed.widen.forward = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.gate = mock
        inp = pt.ones(3, 1)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Sigmoid.forward', return_value=pt.ones(3, 4))
    def test_return_value(self, gate):
        project = Mock(return_value=pt.ones(3, 4))
        expand = Mock(return_value=pt.ones(3, 8))
        self.embed.project.forward = project
        self.embed.widen.forward = expand
        inp = pt.ones(3, 1)
        actual = self.embed(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)


class TestUsageMultiFeature(unittest.TestCase):

    def setUp(self):
        self.embed = GatedResidualEmbedder(
            4,
            identity,
            identity,
            inp_dim=2,
            bias=False
        )
        self.embed.project.weight.data = pt.ones(4, 2) / 2
        self.embed.widen.weight.data = pt.ones(8, 4) / 4

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 2)
        actual = self.embed(inp)
        expected = pt.ones(1, 2, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2)
        actual = self.embed(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_project_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.project.forward = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        mock.assert_called_once_with(inp)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(3, 4))
        self.embed.activate = mock
        inp = pt.ones(3, 2)
        _ = self.embed(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

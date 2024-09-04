import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Linear, Sigmoid, Dropout, ELU, GELU, AlphaDropout
from swak.pt.misc import identity
from swak.pt.mix import GatedResidualConcatMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualConcatMixer(4, 3)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.mix, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.mix.mod_dim, int)
        self.assertEqual(4, self.mix.mod_dim)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(3, self.mix.n_features)

    def test_has_activate(self):
        self.assertTrue(hasattr(self.mix, 'activate'))

    def test_activate(self):
        self.assertIsInstance(self.mix.activate, ELU)

    def test_has_gate(self):
        self.assertTrue(hasattr(self.mix, 'gate'))

    def test_gate(self):
        self.assertIsInstance(self.mix.gate, Sigmoid)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.mix, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.mix.drop, Dropout)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.mix, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedResidualConcatMixer(4, 3)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((12, 4), args_1[0])
        self.assertTupleEqual((4, 8), args_2[0])

    def test_has_project(self):
        self.assertTrue(hasattr(self.mix, 'project'))

    def test_project(self):
        self.assertIsInstance(self.mix.project, Linear)
        self.assertEqual(12, self.mix.project.in_features)
        self.assertEqual(4, self.mix.project.out_features)

    def test_has_expand(self):
        self.assertTrue(hasattr(self.mix, 'expand'))

    def test_expand(self):
        self.assertIsInstance(self.mix.expand, Linear)
        self.assertEqual(4, self.mix.expand.in_features)
        self.assertEqual(8, self.mix.expand.out_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.mix.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, GatedResidualConcatMixer)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertIs(self.mix.activate, new.activate)
        self.assertIs(self.mix.gate, new.gate)
        self.assertIs(self.mix.drop, new.drop)
        self.assertDictEqual(self.mix.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualConcatMixer(
            4,
            2,
            GELU(),
            ELU(),
            AlphaDropout(),
            bias=False
        )

    def test_activate(self):
        self.assertIsInstance(self.mix.activate, GELU)

    def test_gate(self):
        self.assertIsInstance(self.mix.gate, ELU)

    def test_drop(self):
        self.assertIsInstance(self.mix.drop, AlphaDropout)

    def test_kwargs(self):
        self.assertDictEqual({'bias': False}, self.mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedResidualConcatMixer(4, 2, bias=False)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((8, 4), args_1[0])
        self.assertTupleEqual((4, 8), args_2[0])
        self.assertDictEqual({'bias': False}, args_1[1])
        self.assertDictEqual({'bias': False}, args_2[1])

    def test_call_new(self):
        new = self.mix.new(8, 3, GELU(), ELU(), AlphaDropout(), bias=False)
        self.assertIsInstance(new, GatedResidualConcatMixer)
        self.assertEqual(8, new.mod_dim)
        self.assertEqual(3, new.n_features)
        self.assertIsInstance(self.mix.activate, GELU)
        self.assertIsInstance(self.mix.gate, ELU)
        self.assertIsInstance(self.mix.drop, AlphaDropout)
        self.assertDictEqual({'bias': False}, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualConcatMixer(
            4,
            2,
            identity,
            identity,
            bias=False
        )
        self.mix.project.weight.data = pt.ones(4, 8) / 8
        self.mix.expand.weight.data = pt.ones(8, 4) / 4

    def test_callable(self):
        self.assertTrue(callable(self.mix))

    def test_1d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_2d(self):
        inp = pt.ones(5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 5, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(1, 3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(1, 3, 5, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_no_features(self):
        mix = GatedResidualConcatMixer(4, 0, bias=False)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_project_called(self):
        mock = Mock(return_value=pt.ones(5, 4))
        self.mix.project.forward = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 8)
        pt.testing.assert_close(actual, expected)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(5, 4))
        self.mix.activate = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward', return_value=pt.ones(5, 4))
    def test_drop_called(self, drop):
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = drop.call_args[0][0]
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)

    def test_expand_called(self):
        mock = Mock(return_value=pt.ones(5, 8))
        self.mix.expand.forward = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(5, 4))
        self.mix.gate = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, Mock
import torch as pt
from torch.nn import Linear, Softmax, GELU, Sigmoid, ELU, Dropout, AlphaDropout
from torch.nn import PReLU
from swak.pt.misc import identity
from swak.pt.mix.weighted import GatedResidualSumMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualSumMixer(4, 3)

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
        _ = GatedResidualSumMixer(4, 3)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((12, 3), args_1[0])
        self.assertTupleEqual((3, 6), args_2[0])

    def test_has_project(self):
        self.assertTrue(hasattr(self.mix, 'project'))

    def test_project(self):
        self.assertIsInstance(self.mix.project, Linear)
        self.assertEqual(12, self.mix.project.in_features)
        self.assertEqual(3, self.mix.project.out_features)

    def test_has_widen(self):
        self.assertTrue(hasattr(self.mix, 'widen'))

    def test_widen(self):
        self.assertIsInstance(self.mix.widen, Linear)
        self.assertEqual(3, self.mix.widen.in_features)
        self.assertEqual(6, self.mix.widen.out_features)

    def test_has_norm(self):
        self.assertTrue(hasattr(self.mix, 'norm'))

    def test_norm(self):
        self.assertIsInstance(self.mix.norm, Softmax)

    def test_has_importance(self):
        self.assertTrue(hasattr(self.mix, 'importance'))

    def test_importance(self):
        self.assertTrue(callable(self.mix.importance))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called_on_instantiation(self, linear):
        activate = PReLU()
        gate = PReLU()
        with patch('torch.nn.PReLU.reset_parameters') as mock:
            _ = GatedResidualSumMixer(4, 3, activate, gate)
            self.assertEqual(2, mock.call_count)
            self.assertEqual(2, linear.call_count)

    @patch('torch.nn.Linear.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.mix.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_reset_parameters_called_on_activations(self):
        mix = GatedResidualSumMixer(4, 3, PReLU(), PReLU())
        with patch('torch.nn.PReLU.reset_parameters') as activation:
            mix.reset_parameters()
            self.assertEqual(2, activation.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, GatedResidualSumMixer)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertIs(self.mix.activate, new.activate)
        self.assertIs(self.mix.gate, new.gate)
        self.assertIs(self.mix.drop, new.drop)
        self.assertDictEqual(self.mix.kwargs, new.kwargs)


class TestAttributes(unittest.TestCase):

    def test_activate(self):
        activate = GELU()
        mix = GatedResidualSumMixer(4, 3, activate)
        self.assertIs(mix.activate, activate)

    def test_gate(self):
        gate = GELU()
        mix = GatedResidualSumMixer(4, 3, gate=gate)
        self.assertIs(mix.gate, gate)

    def test_drop(self):
        drop = AlphaDropout()
        mix = GatedResidualSumMixer(4, 3, drop=drop)
        self.assertIs(mix.drop, drop)

    def test_kwargs(self):
        mix = GatedResidualSumMixer(4, 3, bias=False)
        self.assertDictEqual({'bias': False}, mix.kwargs)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = GatedResidualSumMixer(4, 2, bias=False)
        args_1, args_2 = mock.call_args_list
        self.assertTupleEqual((8, 2), args_1[0])
        self.assertTupleEqual((2, 4), args_2[0])
        self.assertDictEqual({'bias': False}, args_1[1])
        self.assertDictEqual({'bias': False}, args_2[1])

    def test_new_called(self):
        activate = GELU()
        gate = GELU()
        drop = AlphaDropout()
        mix = GatedResidualSumMixer(4, 3)
        new = mix.new(5, 4, activate, gate, drop, bias=False)
        self.assertEqual(5, new.mod_dim)
        self.assertEqual(4, new.n_features)
        self.assertIs(new.activate, activate)
        self.assertIs(new.gate, gate)
        self.assertIs(new.drop, drop)
        self.assertDictEqual({'bias': False}, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualSumMixer(
            4,
            2,
            identity,
            identity,
            bias=False
        )
        self.mix.project.weight.data = pt.ones(2, 8)
        self.mix.widen.weight.data = pt.ones(4, 2)

    def test_callable(self):
        self.assertTrue(callable(self.mix))

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 3, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_no_features(self):
        mix = GatedResidualSumMixer(4, 0)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = GatedResidualSumMixer(4, 1, bias=False)
        mix.project.weight.data = pt.ones(1, 4)
        mix.widen.weight.data = pt.ones(2, 1)
        inp = pt.ones(3, 1, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GatedResidualSumMixer(4, 5, bias=False)
        mix.project.weight.data = pt.ones(5, 20)
        mix.widen.weight.data = pt.ones(10, 5)
        inp = pt.ones(3, 5, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_project_called(self):
        mock = Mock(return_value=pt.ones(5, 2))
        self.mix.project.forward = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 8)
        pt.testing.assert_close(actual, expected)

    def test_activate_called(self):
        mock = Mock(return_value=pt.ones(5, 2))
        self.mix.activate = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 2) * 8
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward', return_value=pt.ones(5, 2))
    def test_drop_called(self, drop):
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = drop.call_args[0][0]
        expected = pt.ones(5, 2) * 8
        pt.testing.assert_close(actual, expected)

    def test_widen_called(self):
        mock = Mock(return_value=pt.ones(5, 4))
        self.mix.widen.forward = mock
        inp = pt.ones(5, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(5, 2) * 8
        pt.testing.assert_close(actual, expected)

    def test_gate_called(self):
        mock = Mock(return_value=pt.ones(3, 2))
        self.mix.gate = mock
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2) * 16
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Softmax.forward')
    def test_norm_called(self, mock):
        mock.return_value = pt.ones(3, 2) / 2
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2) * (16 * 16 + 8)
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = GatedResidualSumMixer(4, 2, bias=False)
        self.mix.project.weight.data = pt.ones(2, 8)
        self.mix.widen.weight.data = pt.ones(4, 2)

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(3, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 3, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 0, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = GatedResidualSumMixer(4, 1, bias=False)
        mix.project.weight.data = pt.ones(1, 4)
        mix.widen.weight.data = pt.ones(2, 1)
        inp = pt.ones(3, 1, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 1)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GatedResidualSumMixer(4, 5, bias=False)
        mix.project.weight.data = pt.ones(5, 20)
        mix.widen.weight.data = pt.ones(10, 5)
        inp = pt.ones(3, 5, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 5) * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

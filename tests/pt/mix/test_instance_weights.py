import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.mix import InstanceWeightsMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = InstanceWeightsMixer(4, 3)

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

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.mix, 'dropout'))

    def test_dropout(self):
        self.assertIsInstance(self.mix.dropout, float)
        self.assertEqual(0.0, self.mix.dropout)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.mix, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.mix.skip, bool)
        self.assertTrue(self.mix.skip)

    def test_has_keep_dim(self):
        self.assertTrue(hasattr(self.mix, 'keep_dim'))

    def test_keep_dim(self):
        self.assertIsInstance(self.mix.keep_dim, bool)
        self.assertFalse(self.mix.keep_dim)

    def test_has_score(self):
        self.assertTrue(hasattr(self.mix, 'score'))

    def test_score(self):
        self.assertIsInstance(self.mix.score, pt.nn.Softmax)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.mix, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.mix.drop, pt.nn.Dropout)
        self.assertEqual(0.0, self.mix.drop.p)

    def test_has_device(self):
        self.assertTrue(hasattr(self.mix, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.mix.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.mix, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.mix.dtype, pt.float)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = InstanceWeightsMixer(4, 3)
        mock.assert_called_once_with(
            in_features=12,
            out_features=3,
            bias=True,
            device='cpu',
            dtype=pt.float
        )

    def test_has_project(self):
        self.assertTrue(hasattr(self.mix, 'project'))

    def test_project(self):
        self.assertIsInstance(self.mix.project, pt.nn.Linear)
        self.assertEqual(12, self.mix.project.in_features)
        self.assertEqual(3, self.mix.project.out_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_reset_parameters_called(self):
        with patch.object(self.mix.project, 'reset_parameters') as mock:
            self.mix.reset_parameters()
            mock.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, InstanceWeightsMixer)
        self.assertIsNot(new, self.mix)
        self.assertIsNot(new.project.weight, self.mix.project.weight)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertEqual(self.mix.dropout, new.dropout)
        self.assertEqual(self.mix.skip, new.skip)
        self.assertEqual(self.mix.keep_dim, new.keep_dim)
        self.assertEqual(self.mix.device, new.device)
        self.assertEqual(self.mix.dtype, new.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = InstanceWeightsMixer(
            mod_dim=4,
            n_features=3,
            bias=False,
            dropout=0.1,
            skip=False,
            keep_dim=True,
            dtype=pt.float16
        )

    def test_bias(self):
        self.assertIsInstance(self.mix.bias, bool)
        self.assertFalse(self.mix.bias)

    def test_dropout(self):
        self.assertIsInstance(self.mix.dropout, float)
        self.assertEqual(0.1, self.mix.dropout)

    def test_drop(self):
        self.assertIsInstance(self.mix.drop, pt.nn.Dropout)
        self.assertEqual(0.1, self.mix.drop.p)

    def test_skip(self):
        self.assertIsInstance(self.mix.skip, bool)
        self.assertFalse(self.mix.skip)

    def test_keep_dim(self):
        self.assertIsInstance(self.mix.keep_dim, bool)
        self.assertTrue(self.mix.keep_dim)

    def test_dtype(self):
        self.assertIs(self.mix.dtype, pt.float16)
        mix = self.mix.to(pt.float64)
        self.assertIs(mix.dtype, pt.float64)

    @patch('torch.nn.Linear')
    def test_linear_called(self, mock):
        _ = InstanceWeightsMixer(4, 3, bias=False, dtype=pt.float16)
        mock.assert_called_once_with(
            in_features=12,
            out_features=3,
            bias=False,
            device='cpu',
            dtype=pt.float16
        )


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = InstanceWeightsMixer(4, 2, bias=False, skip=False)
        self.mix.project.weight.data = pt.ones(2, 8)

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4)
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 4)
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 5, 4)
        pt.testing.assert_close(actual, expected)

    def test_5d(self):
        inp = pt.ones(1, 3, 5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(1, 3, 5, 4)
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(3, 0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_skip(self):
        mix = InstanceWeightsMixer(4, 2, bias=False, skip=True)
        mix.project.weight.data = pt.ones(2, 8)
        inp = pt.ones(3, 2, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_keep_dim(self):
        mix = InstanceWeightsMixer(4, 2, bias=False, keep_dim=True)
        mix.project.weight.data = pt.ones(2, 8)
        inp = pt.ones(3, 2, 4)
        actual = mix(inp)
        expected = pt.ones(3, 1, 4) * 2
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_false(self):
        mix = InstanceWeightsMixer(4, 0, skip=False, keep_dim=False)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_true(self):
        mix = InstanceWeightsMixer(4, 0, skip=False, keep_dim=True)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_false(self):
        mix = InstanceWeightsMixer(4, 0, skip=True, keep_dim=False)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_true(self):
        mix = InstanceWeightsMixer(4, 0, skip=True, keep_dim=True)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Linear.forward', return_value=pt.ones(3, 2))
    def test_linear_called(self, mock):
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 8)
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Softmax.forward')
    def test_norm_called(self, mock):
        mock.return_value = pt.ones(3, 2)
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2) * 8
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward')
    def test_drop_called(self, mock):
        mock.return_value = pt.ones(3, 2)
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 1, 4)
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = InstanceWeightsMixer(4, 2, bias=False)
        self.mix.project.weight.data = pt.ones(2, 8, device='cpu')

    def test_2d(self):
        inp = pt.ones(2, 4, device='cpu')
        actual = self.mix.importance(inp)
        expected = pt.ones(2, device='cpu') * 0.5
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = self.mix.importance(inp)
        expected = pt.ones(3, 2, device='cpu') * 0.5
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4, device='cpu')
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 3, 2, device='cpu') * 0.5
        pt.testing.assert_close(actual, expected)

    def test_accepts_mask(self):
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = self.mix.importance(inp, mask='foo')
        expected = pt.ones(3, 2, device='cpu') * 0.5
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4, device='cpu')
        actual = self.mix.importance(inp)
        expected = pt.ones(5, 0, 2, device='cpu') * 0.5
        pt.testing.assert_close(actual, expected)

    def test_no_feature(self):
        mix = InstanceWeightsMixer(4, 0)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 0, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = InstanceWeightsMixer(4, 1, bias=False)
        mix.project.weight.data = pt.ones(1, 4, device='cpu')
        inp = pt.ones(3, 1, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 1, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = InstanceWeightsMixer(4, 5, bias=False)
        mix.project.weight.data = pt.ones(5, 20, device='cpu')
        inp = pt.ones(3, 5, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 5, device='cpu') * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

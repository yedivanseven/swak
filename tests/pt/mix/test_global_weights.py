import unittest
from unittest.mock import patch
import torch as pt
from torch.nn import Softmax
from swak.pt.mix import GlobalWeightsMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GlobalWeightsMixer(4, 3)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.mix, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.mix.mod_dim, int)
        self.assertEqual(4, self.mix.mod_dim)

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

    def test_has_device(self):
        self.assertTrue(hasattr(self.mix, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.mix.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.mix, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.mix.dtype, pt.float)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.mix, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.mix.n_features, int)
        self.assertEqual(3, self.mix.n_features)

    def test_has_coeffs(self):
        self.assertTrue(hasattr(self.mix, 'coeffs'))

    def test_coeffs(self):
        expected = pt.ones(3, device='cpu')
        pt.testing.assert_close(self.mix.coeffs, expected)

    def test_has_score(self):
        self.assertTrue(hasattr(self.mix, 'score'))

    def test_score(self):
        self.assertIsInstance(self.mix.score, Softmax)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.mix, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.mix.drop, pt.nn.Dropout)
        self.assertEqual(0.0, self.mix.drop.p)

    def test_has_importance(self):
        self.assertTrue(hasattr(self.mix, 'importance'))

    def test_importance(self):
        self.assertTrue(callable(self.mix.importance))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_call_reset_parameters(self):
        mix = GlobalWeightsMixer(4, 3)
        mix.coeffs.data.fill_(2.0)
        expected = pt.ones(3, device='cpu') * 2
        pt.testing.assert_close(mix.coeffs, expected)

        mix.reset_parameters()
        expected = pt.ones(3, device='cpu')
        pt.testing.assert_close(mix.coeffs, expected)

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, GlobalWeightsMixer)
        self.assertIsNot(new, self.mix)
        self.assertIsNot(new.coeffs, self.mix.coeffs)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_features, new.n_features)
        self.assertEqual(self.mix.dropout, new.dropout)
        self.assertEqual(self.mix.skip, new.skip)
        self.assertEqual(self.mix.keep_dim, new.keep_dim)
        self.assertEqual(self.mix.device, new.device)
        self.assertEqual(self.mix.dtype, new.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = GlobalWeightsMixer(
            mod_dim=4,
            n_features=3,
            dropout=0.1,
            skip=False,
            keep_dim=True,
            dtype=pt.float16
        )

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


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = GlobalWeightsMixer(4, 2, skip=False)

    def test_2d(self):
        inp = pt.ones(2, 4, device='cpu')
        actual = self.mix(inp)
        expected = pt.ones(4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = self.mix(inp)
        expected = pt.ones(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(5, 3, 2, 4, device='cpu')
        actual = self.mix(inp)
        expected = pt.ones(5, 3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(5, 0, 2, 4, device='cpu')
        actual = self.mix(inp)
        expected = pt.ones(5, 0, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_accepts_mask(self):
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = self.mix(inp, mask='foo')
        expected = pt.ones(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_skip(self):
        mix = GlobalWeightsMixer(4, 2, skip=True)
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = mix(inp)
        expected = pt.ones(3, 4, device='cpu') * 2
        pt.testing.assert_close(actual, expected)

    def test_keep_dim(self):
        mix = GlobalWeightsMixer(4, 2, keep_dim=True)
        inp = pt.ones(3, 2, 4, device='cpu')
        actual = mix(inp)
        expected = pt.ones(3, 1, 4, device='cpu') * 2
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_false(self):
        mix = GlobalWeightsMixer(4, 0, skip=False, keep_dim=False)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_true(self):
        mix = GlobalWeightsMixer(4, 0, skip=False, keep_dim=True)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_false(self):
        mix = GlobalWeightsMixer(4, 0, skip=True, keep_dim=False)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_true(self):
        mix = GlobalWeightsMixer(4, 0, skip=True, keep_dim=True)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = GlobalWeightsMixer(4, 1, skip=False)
        inp = pt.ones(3, 1, 4, device='cpu')
        actual = mix(inp)
        expected = pt.ones(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GlobalWeightsMixer(4, 5, skip=False)
        inp = pt.ones(3, 5, 4, device='cpu')
        actual = mix(inp)
        expected = pt.ones(3, 4, device='cpu')
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Softmax.forward')
    def test_norm_called(self, mock):
        mock.return_value = pt.ones(3, 2, device='cpu')
        inp = pt.ones(3, 2, 4, device='cpu')
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 2, device='cpu')
        pt.testing.assert_close(actual, expected)

    @patch('torch.nn.Dropout.forward')
    def test_drop_called(self, mock):
        mock.return_value = pt.ones(3, 2, device='cpu')
        inp = pt.ones(3, 2, 4, device='cpu')
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 1, 4, device='cpu')
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = GlobalWeightsMixer(4, 2)

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
        mix = GlobalWeightsMixer(4, 0)
        inp = pt.ones(3, 0, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 0, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        mix = GlobalWeightsMixer(4, 1)
        inp = pt.ones(3, 1, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 1, device='cpu')
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        mix = GlobalWeightsMixer(4, 5)
        inp = pt.ones(3, 5, 4, device='cpu')
        actual = mix.importance(inp)
        expected = pt.ones(3, 5, device='cpu') * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

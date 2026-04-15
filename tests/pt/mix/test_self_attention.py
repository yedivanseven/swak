import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.mix import SelfAttentionMixer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = SelfAttentionMixer(4)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.mix, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.mix.mod_dim, int)
        self.assertEqual(4, self.mix.mod_dim)

    def test_has_n_heads(self):
        self.assertTrue(hasattr(self.mix, 'n_heads'))

    def test_n_heads(self):
        self.assertIsInstance(self.mix.n_heads, int)
        self.assertEqual(1, self.mix.n_heads)

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

    def test_has_attention(self):
        self.assertTrue(hasattr(self.mix, 'attention'))

    def test_attention(self):
        self.assertIsInstance(self.mix.attention, pt.nn.MultiheadAttention)

    @patch('torch.nn.MultiheadAttention')
    def test_attention_called(self, mock):
        _ = SelfAttentionMixer(4)
        mock.assert_called_once_with(
            embed_dim=4,
            num_heads=1,
            dropout=0.0,
            bias=True,
            batch_first=True,
            device='cpu',
            dtype=pt.float
        )

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.mix, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.mix.reset_parameters))

    def test_reset_parameters_called(self):
        with patch.object(self.mix.attention, '_reset_parameters') as mock:
            self.mix.reset_parameters()
            mock.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.mix, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.mix.new))

    def test_call_new(self):
        new = self.mix.new()
        self.assertIsInstance(new, SelfAttentionMixer)
        self.assertIsNot(new, self.mix)
        self.assertIsNot(new.attention, self.mix.attention)
        self.assertEqual(self.mix.mod_dim, new.mod_dim)
        self.assertEqual(self.mix.n_heads, new.n_heads)
        self.assertEqual(self.mix.dropout, new.dropout)
        self.assertEqual(self.mix.skip, new.skip)
        self.assertEqual(self.mix.keep_dim, new.keep_dim)
        self.assertEqual(self.mix.device, new.device)
        self.assertEqual(self.mix.dtype, new.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mix = SelfAttentionMixer(
            mod_dim=4,
            n_heads=2,
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

    @patch('torch.nn.MultiheadAttention')
    def test_attention_called(self, mock):
        _ = SelfAttentionMixer(
            mod_dim=4,
            n_heads=2,
            bias=False,
            dropout=0.1,
            skip=False,
            keep_dim=True,
            dtype=pt.float16
        )
        mock.assert_called_once_with(
            embed_dim=4,
            num_heads=2,
            dropout=0.1,
            bias=False,
            batch_first=True,
            device='cpu',
            dtype=pt.float16
        )

class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mix = SelfAttentionMixer(4, bias=False, skip=False)
        self.mix.attention.in_proj_weight.data = pt.ones(12, 4)
        self.mix.attention.out_proj.weight.data = pt.ones(4, 4)

    def test_2d(self):
        inp = pt.ones(2, 4)
        actual = self.mix(inp)
        expected = pt.ones(4) * 16
        pt.testing.assert_close(actual, expected)

    def test_3d(self):
        inp = pt.ones(5, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(5, 4) * 16
        pt.testing.assert_close(actual, expected)

    def test_4d(self):
        inp = pt.ones(3, 5, 2, 4)
        with self.assertRaises(AssertionError):
            _ = self.mix(inp)

    def test_empty_dims(self):
        inp = pt.ones(0, 2, 4)
        actual = self.mix(inp)
        expected = pt.ones(0, 4)
        pt.testing.assert_close(actual, expected)

    def test_skip(self):
        mix = SelfAttentionMixer(4, bias=False, skip=True)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 2, 4)
        actual = mix(inp)
        expected = pt.ones(3, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_keep_dim(self):
        mix = SelfAttentionMixer(4, bias=False, keep_dim=True)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 2, 4)
        actual = mix(inp)
        expected = pt.ones(3, 1, 4) * 16 + 1
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_false(self):
        mix = SelfAttentionMixer(4, skip=False, keep_dim=False, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_false_keep_dim_true(self):
        mix = SelfAttentionMixer(4, skip=False, keep_dim=True, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_false(self):
        mix = SelfAttentionMixer(4, skip=True, keep_dim=False, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 4)
        pt.testing.assert_close(actual, expected)

    def test_no_features_skip_true_keep_dim_true(self):
        mix = SelfAttentionMixer(4, skip=True, keep_dim=True, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 0, 4)
        actual = mix(inp)
        expected = pt.zeros(3, 0, 4)
        pt.testing.assert_close(actual, expected)

    @patch(
        'torch.nn.MultiheadAttention.forward',
        return_value=(pt.ones(3, 2), None)
    )
    def test_attention_called(self, mock):
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp, mask='mask')
        query, key, value = mock.call_args[0]
        pt.testing.assert_close(query, inp)
        pt.testing.assert_close(key, inp)
        pt.testing.assert_close(value, inp)
        self.assertDictEqual(
        {'key_padding_mask': 'mask', 'need_weights': False},
        mock.call_args[1]
        )

    @patch('torch.nn.Dropout.forward')
    def test_drop_called(self, mock):
        mix = SelfAttentionMixer(4, skip=True, keep_dim=True, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        mock.return_value = pt.ones(3, 2)
        inp = pt.ones(3, 2, 4)
        _ = self.mix(inp)
        actual = mock.call_args[0][0]
        expected = pt.ones(3, 1, 4) * 16
        pt.testing.assert_close(actual, expected)


class TestImportance(unittest.TestCase):

    def setUp(self):
        self.mix = SelfAttentionMixer(4, 2, bias=False)
        self.mix.attention.in_proj_weight.data = pt.ones(12, 4)
        self.mix.attention.out_proj.weight.data = pt.ones(4, 4)

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

    def test_accepts_mask(self):
        inp = pt.ones(3, 2, 4)
        mask = pt.tensor([
            [True, False],
            [False, True],
            [True, False]
        ])
        actual = self.mix.importance(inp, mask=mask)
        expected = pt.tensor([
            [0., 1.],
            [1., 0.],
            [0., 1.]
        ])
        pt.testing.assert_close(actual, expected)

    def test_empty_dims(self):
        inp = pt.ones(0, 2, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(0, 2) * 0.5
        pt.testing.assert_close(actual, expected)

    def test_no_feature(self):
        mix = SelfAttentionMixer(4, skip=True, bias=False)
        mix.attention.in_proj_weight.data = pt.ones(12, 4)
        mix.attention.out_proj.weight.data = pt.ones(4, 4)
        inp = pt.ones(3, 0, 4)
        actual = mix.importance(inp)
        expected = pt.ones(3, 0)
        pt.testing.assert_close(actual, expected)

    def test_1_feature(self):
        inp = pt.ones(3, 1, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(3, 1)
        pt.testing.assert_close(actual, expected)

    def test_5_features(self):
        inp = pt.ones(3, 5, 4)
        actual = self.mix.importance(inp)
        expected = pt.ones(3, 5) * 0.2
        pt.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

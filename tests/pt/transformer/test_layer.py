import sys
import unittest
from unittest.mock import patch, Mock
import torch as pt
from swak.pt.blocks import ActivatedBlock, IdentityBlock
from swak.pt.transformer import (
    MultiheadedSelfAttention,
    EncoderLayer,
    Sinusoidal
)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(self.attention, self.feedforward)

    def test_has_attention(self):
        self.assertTrue(hasattr(self.layer, 'attention'))

    def test_attention(self):
        self.assertIs(self.layer.attention, self.attention)

    def test_has_feed_forward(self):
        self.assertTrue(hasattr(self.layer, 'feed_forward'))

    def test_feed_forward(self):
        self.assertIs(self.layer.feed_forward, self.feedforward)

    def test_has_pos_enc(self):
        self.assertTrue(hasattr(self.layer, 'pos_enc'))

    def test_pos_enc(self):
        self.assertIsInstance(self.layer.pos_enc, IdentityBlock)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.layer, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.layer.bias, bool)
        self.assertTrue(self.layer.bias)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.layer, 'dropout'))

    def test_dropout(self):
        self.assertEqual(0.1, self.layer.dropout)

    def test_has_norm_cls(self):
        self.assertTrue(hasattr(self.layer, 'norm_cls'))

    def test_norm_cls(self):
        self.assertIs(self.layer.norm_cls, pt.nn.LayerNorm)

    def test_has_norm_first(self):
        self.assertTrue(hasattr(self.layer, 'norm_first'))

    def test_norm_first(self):
        self.assertIsInstance(self.layer.norm_first, bool)
        self.assertTrue(self.layer.norm_first)

    def test_has_eps(self):
        self.assertTrue(hasattr(self.layer, 'eps'))

    def test_eps(self):
        self.assertEqual(1e-5, self.layer.eps)

    def test_has_device(self):
        self.assertTrue(hasattr(self.layer, 'device'))

    def test_device(self):
        self.assertIsInstance(self.layer.device, pt.device)
        self.assertEqual(self.attention.device, self.layer.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.layer, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.layer.attention.dtype, self.layer.dtype)

    def test_has_norm1(self):
        self.assertTrue(hasattr(self.layer, 'norm1'))

    def test_norm1(self):
        self.assertIsInstance(self.layer.norm1, pt.nn.LayerNorm)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm1.normalized_shape
        )
        self.assertEqual(self.layer.eps, self.layer.norm1.eps)
        self.assertTrue(self.layer.norm1.elementwise_affine)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm1.bias.shape
        )
        self.assertEqual(self.layer.device, self.layer.norm1.weight.device)
        self.assertEqual(self.layer.dtype, self.layer.norm1.weight.dtype)

    def test_has_norm2(self):
        self.assertTrue(hasattr(self.layer, 'norm2'))

    def test_norm2(self):
        self.assertIsInstance(self.layer.norm2, pt.nn.LayerNorm)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm2.normalized_shape
        )
        self.assertEqual(self.layer.eps, self.layer.norm2.eps)
        self.assertTrue(self.layer.norm2.elementwise_affine)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm2.bias.shape
        )
        self.assertEqual(self.layer.device, self.layer.norm2.weight.device)
        self.assertEqual(self.layer.dtype, self.layer.norm2.weight.dtype)

    def test_has_drop1(self):
        self.assertTrue(hasattr(self.layer, 'drop1'))

    def test_drop1(self):
        self.assertIsInstance(self.layer.drop1, pt.nn.Dropout)
        self.assertEqual(self.layer.dropout, self.layer.drop1.p)

    def test_has_drop2(self):
        self.assertTrue(hasattr(self.layer, 'drop2'))

    def test_drop2(self):
        self.assertIsInstance(self.layer.drop2, pt.nn.Dropout)
        self.assertEqual(self.layer.dropout, self.layer.drop2.p)

    def test_has_bias_kwarg(self):
        self.assertTrue(hasattr(self.layer, 'bias_kwarg'))

    def test_bias_kwarg(self):
        self.assertDictEqual({'bias': True}, self.layer.bias_kwarg)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.layer, 'mod_dim'))

    def test_mod_dim(self):
        self.assertEqual(self.attention.mod_dim, self.layer.mod_dim)

    def test_has_has_pos_enc(self):
        self.assertTrue(hasattr(self.layer, 'has_pos_enc'))

    def test_has_pos_enc_correct(self):
        self.assertIsInstance(self.layer.has_pos_enc, bool)
        self.assertFalse(self.layer.has_pos_enc)
        pos_enc = Sinusoidal(self.mod_dim, 128)
        layer = EncoderLayer(self.attention, self.feedforward, pos_enc=pos_enc)
        self.assertTrue(layer.has_pos_enc)

    def test_has_context(self):
        self.assertTrue(hasattr(self.layer, 'context'))

    def test_context(self):
        self.assertIsInstance(self.layer.context, int)
        self.assertEqual(sys.maxsize, self.layer.context)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.layer, 'reset_parameters'))

    def test_reset_parameters_callable(self):
        self.assertTrue(callable(self.layer.reset_parameters))

    @patch('torch.nn.LayerNorm.reset_parameters')
    def test_call_reset_parameters(self, norm):
        attention = Mock()
        attention.mod_dim = self.mod_dim
        attention.device = self.attention.device
        attention.dtype = self.attention.dtype
        attention.to.return_value = attention
        feedforward = Mock()
        feedforward.to.return_value = feedforward
        pos_enc = Mock()
        pos_enc.to.return_value = pos_enc
        layer = EncoderLayer(attention, feedforward, pos_enc)
        self.assertEqual(2, norm.call_count)
        layer.reset_parameters()
        self.assertEqual(4, norm.call_count)
        attention.reset_parameters.assert_called_once_with()
        feedforward.reset_parameters.assert_called_once_with()
        pos_enc.reset_parameters.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.layer, 'new'))

    def test_new_callable(self):
        self.assertTrue(callable(self.layer.new))

    def test_call_new(self):
        new = self.layer.new()
        self.assertIsInstance(new, EncoderLayer)
        self.assertIsNot(new, self.layer)
        self.assertIsInstance(new.attention, MultiheadedSelfAttention)
        self.assertIsNot(new.attention, self.layer.attention)
        self.assertIsInstance(new.feed_forward, ActivatedBlock)
        self.assertIsNot(new.feed_forward, self.layer.feed_forward)
        self.assertIsInstance(new.pos_enc, IdentityBlock)
        self.assertIsNot(new.pos_enc, self.layer.pos_enc)
        self.assertEqual(new.bias, self.layer.bias)
        self.assertEqual(new.dropout, self.layer.dropout)
        self.assertEqual(new.norm_first, self.layer.norm_first)
        self.assertEqual(new.eps, self.layer.eps)
        self.assertEqual(new.device, self.layer.device)
        self.assertEqual(new.dtype, self.layer.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 128
        self.dtype = pt.double
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc,
            False,
            0.2,
            False,
            pt.nn.RMSNorm,
            1e-4,
            dtype=self.dtype
        )

    def test_pos_enc(self):
        self.assertIsInstance(self.layer.pos_enc, Sinusoidal)

    def test_bias(self):
        self.assertFalse(self.layer.bias)

    def test_dropout(self):
        self.assertEqual(0.2, self.layer.dropout)

    def test_norm_cls(self):
        self.assertIs(self.layer.norm_cls, pt.nn.RMSNorm)

    def test_norm1(self):
        self.assertIsInstance(self.layer.norm1, pt.nn.RMSNorm)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm1.normalized_shape
        )
        self.assertEqual(self.layer.eps, self.layer.norm1.eps)
        self.assertTrue(self.layer.norm1.elementwise_affine)
        self.assertEqual(self.layer.device, self.layer.norm1.weight.device)
        self.assertEqual(self.layer.dtype, self.layer.norm1.weight.dtype)

    def test_norm2(self):
        self.assertIsInstance(self.layer.norm2, pt.nn.RMSNorm)
        self.assertTupleEqual(
            (self.attention.mod_dim,),
            self.layer.norm2.normalized_shape
        )
        self.assertEqual(self.layer.eps, self.layer.norm2.eps)
        self.assertTrue(self.layer.norm2.elementwise_affine)
        self.assertEqual(self.layer.device, self.layer.norm2.weight.device)
        self.assertEqual(self.layer.dtype, self.layer.norm2.weight.dtype)

    def test_bias_kwarg(self):
        self.assertDictEqual({}, self.layer.bias_kwarg)

    def test_norm_first(self):
        self.assertFalse(self.layer.norm_first)

    def test_eps(self):
        self.assertEqual(1e-4, self.layer.eps),

    def test_dtype(self):
        self.assertIs(self.layer.dtype, self.dtype)
        self.assertIs(self.attention.dtype, self.dtype)
        self.assertIs(self.pos_enc.dtype, self.dtype)
        self.assertIs(self.layer.attention.dtype, self.dtype)
        self.assertIs(self.layer.pos_enc.dtype, self.dtype)

    def test_has_pos_enc_correct(self):
        self.assertTrue(self.layer.has_pos_enc)

    def test_double_pos_enc_warns(self):
        attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=self.pos_enc
        )
        with self.assertWarns(UserWarning):
            _ = EncoderLayer(attention, self.feedforward, self.pos_enc)

    def test_context_pos_enc_on_layer(self):
        self.assertEqual(self.context, self.layer.context)

    def test_context_pos_enc_on_attention(self):
        attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=self.pos_enc
        )
        _ = EncoderLayer(attention, self.feedforward)
        self.assertEqual(self.context, self.layer.context)

    def test_context_pos_enc_on_attn_smaller(self):
        pos_enc = Sinusoidal(self.mod_dim, 64)
        attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=pos_enc
        )
        layer = EncoderLayer(attention, self.feedforward, self.pos_enc)
        self.assertEqual(64, layer.context)

    def test_context_pos_enc_on_layer_smaller(self):
        pos_enc = Sinusoidal(self.mod_dim, 64)
        attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=self.pos_enc
        )
        layer = EncoderLayer(attention, self.feedforward, pos_enc)
        self.assertEqual(64, layer.context)

class TestUsageNormFirst(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.pos_enc = Sinusoidal(self.mod_dim, 128)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.inp = pt.rand(5, 32, self.mod_dim, device='cpu')

    def test_pos_enc_called(self):
        pos_enc = Mock()
        pos_enc.return_value = self.inp
        pos_enc.to.return_value = pos_enc
        layer = EncoderLayer(self.attention, self.feedforward, pos_enc)
        _ = layer(self.inp)
        pos_enc.assert_called_once()
        pt.testing.assert_close(pos_enc.call_args[0][0], self.inp)

    def test_norm1_called(self):
        with patch.object(
                self.layer.pos_enc,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.norm1,
            'forward',
                return_value=self.inp
        ) as norm1:
            _ = self.layer(self.inp)
            norm1.assert_called_once()
            pt.testing.assert_close(norm1.call_args[0][0], self.inp)

    def test_attention_called(self):
        with patch.object(
                self.layer.norm1,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.attention,
            'forward',
                return_value=self.inp
        ) as attention:
            mask = object()
            _ = self.layer(self.inp, mask, False)
            attention.assert_called_once()
            pt.testing.assert_close(attention.call_args[0][0], self.inp)
            self.assertIs(mask, attention.call_args[0][1])
            self.assertFalse(attention.call_args[0][2])

    def test_drop1_called(self):
        with patch.object(
                self.layer.attention,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop1,
            'forward',
                return_value=self.inp
        ) as drop1:
            _ = self.layer(self.inp)
            drop1.assert_called_once()
            pt.testing.assert_close(drop1.call_args[0][0], self.inp)

    def test_norm2_called(self):
        with patch.object(
                self.layer.drop1,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.pos_enc,
            'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.norm2,
            'forward',
                return_value=self.inp
        ) as norm2:
            _ = self.layer(self.inp)
            norm2.assert_called_once()
            pt.testing.assert_close(norm2.call_args[0][0], 2 * self.inp)

    def test_feed_forward_called(self):
        with patch.object(
                self.layer.norm2,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.feed_forward,
            'forward',
                return_value=self.inp
        ) as feedforward:
            _ = self.layer(self.inp)
            feedforward.assert_called_once()
            pt.testing.assert_close(feedforward.call_args[0][0], self.inp)

    def test_drop2_called(self):
        with patch.object(
                self.layer.feed_forward,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop2,
            'forward',
                return_value=self.inp
        ) as drop2:
            _ = self.layer(self.inp)
            drop2.assert_called_once()
            pt.testing.assert_close(drop2.call_args[0][0], self.inp)

    def test_output(self):
        with patch.object(
                self.layer.pos_enc,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop1,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop2,
            'forward',
                return_value=self.inp
        ):
            out = self.layer(self.inp)
            pt.testing.assert_close(out, 3 * self.inp)


class TestUsageNormLast(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.pos_enc = Sinusoidal(self.mod_dim, 128)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc,
            norm_first=False
        )
        self.inp = pt.rand(5, 32, self.mod_dim, device='cpu')

    def test_pos_enc_called(self):
        pos_enc = Mock()
        pos_enc.return_value = self.inp
        pos_enc.to.return_value = pos_enc
        layer = EncoderLayer(self.attention, self.feedforward, pos_enc)
        _ = layer(self.inp)
        pos_enc.assert_called_once()
        pt.testing.assert_close(pos_enc.call_args[0][0], self.inp)

    def test_attention_called(self):
        with patch.object(
                self.layer.pos_enc,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.attention,
            'forward',
                return_value=self.inp
        ) as attention:
            mask = object()
            _ = self.layer(self.inp, mask, False)
            attention.assert_called_once()
            pt.testing.assert_close(attention.call_args[0][0], self.inp)
            self.assertIs(mask, attention.call_args[0][1])
            self.assertFalse(attention.call_args[0][2])

    def test_drop1_called(self):
        with patch.object(
                self.layer.attention,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop1,
            'forward',
                return_value=self.inp
        ) as drop1:
            _ = self.layer(self.inp)
            drop1.assert_called_once()
            pt.testing.assert_close(drop1.call_args[0][0], self.inp)

    def test_norm1_called(self):
        with patch.object(
                self.layer.pos_enc,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop1,
                'forward',
                return_value=self.inp
        ),patch.object(
                self.layer.norm1,
            'forward',
                return_value=self.inp
        ) as norm1:
            _ = self.layer(self.inp)
            norm1.assert_called_once()
            pt.testing.assert_close(norm1.call_args[0][0], 2 * self.inp)

    def test_feed_forward_called(self):
        with patch.object(
                self.layer.norm1,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.feed_forward,
            'forward',
                return_value=self.inp
        ) as feedforward:
            _ = self.layer(self.inp)
            feedforward.assert_called_once()
            pt.testing.assert_close(feedforward.call_args[0][0], self.inp)

    def test_drop2_called(self):
        with patch.object(
                self.layer.feed_forward,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop2,
            'forward',
                return_value=self.inp
        ) as drop2:
            _ = self.layer(self.inp)
            drop2.assert_called_once()
            pt.testing.assert_close(drop2.call_args[0][0], self.inp)

    def test_norm2_called(self):
        with patch.object(
                self.layer.norm1,
                'forward',
                return_value=self.inp
        ), patch.object(
                self.layer.drop2,
                'forward',
                return_value=self.inp
        ),patch.object(
                self.layer.norm2,
            'forward',
                return_value=self.inp
        ) as norm2:
            _ = self.layer(self.inp)
            norm2.assert_called_once()
            pt.testing.assert_close(norm2.call_args[0][0], 2 * self.inp)

    def test_output(self):
        with patch.object(
                self.layer.norm2,
                'forward',
                return_value=self.inp
        ):
            out = self.layer(self.inp)
            pt.testing.assert_close(out, self.inp)


class TestUsageOutputShape(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 61
        self.batch_size = 17
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )

    def test_2d(self):
        inp = pt.rand(self.context, self.mod_dim, device='cpu')
        actual = self.attention(inp)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d(self):
        inp = pt.rand(
            self.batch_size,
            self.context,
            self.mod_dim,
            device='cpu'
        )
        actual = self.attention(inp)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_4d(self):
        inp = pt.rand(
            7,
            self.batch_size,
            self.context,
            self.mod_dim,
            device='cpu'
        )
        actual = self.attention(inp)
        self.assertTupleEqual(inp.shape, actual.shape)


if __name__ == '__main__':
    unittest.main()

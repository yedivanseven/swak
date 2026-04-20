import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.blocks import ActivatedBlock, IdentityBlock
from swak.pt.transformer import (
    MultiheadedSelfAttention,
    EncoderLayer,
    Encoder,
    Sinusoidal
)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.encode = Encoder(self.layer)

    def test_has_n_layers(self):
        self.assertTrue(hasattr(self.encode, 'n_layers'))

    def test_n_layers(self):
        self.assertIsInstance(self.encode.n_layers, int)
        self.assertEqual(1, self.encode.n_layers)

    def test_has_pos_enc(self):
        self.assertTrue(hasattr(self.encode, 'pos_enc'))

    def test_pos_enc(self):
        self.assertIsInstance(self.encode.pos_enc, IdentityBlock)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.encode, 'dropout'))

    def test_dropout(self):
        self.assertEqual(0.0, self.encode.dropout)

    def test_has_device(self):
        self.assertTrue(hasattr(self.encode, 'device'))

    def test_device(self):
        self.assertIsInstance(self.encode.device, pt.device)
        self.assertEqual('cpu', self.encode.device.type)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.encode, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.encode.dtype, pt.float)

    def test_has_layers(self):
        self.assertTrue(hasattr(self.encode, 'layers'))

    def test_layers(self):
        self.assertIsInstance(self.encode.layers, pt.nn.ModuleList)
        for layer in self.encode.layers:
            self.assertIsInstance(layer, EncoderLayer)
            self.assertIs(layer.dtype, self.encode.dtype)
            self.assertEqual(layer.device, self.encode.device)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.encode, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.encode.drop, pt.nn.Dropout)
        self.assertEqual(self.encode.dropout, self.encode.drop.p)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.encode, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.encode.mod_dim, int)
        self.assertEqual(self.mod_dim, self.encode.mod_dim)

    def test_has_context(self):
        self.assertTrue(hasattr(self.encode, 'context'))

    def test_context(self):
        self.assertIsInstance(self.encode.context, int)
        self.assertEqual(self.context, self.encode.context)

    def test_has_has_pos_enc(self):
        self.assertTrue(hasattr(self.layer, 'has_pos_enc'))

    def test_has_pos_enc_correct(self):
        self.assertIsInstance(self.layer.has_pos_enc, bool)
        self.assertTrue(self.layer.has_pos_enc)
        pos_enc = IdentityBlock(self.mod_dim)
        layer = EncoderLayer(self.attention, self.feedforward, pos_enc=pos_enc)
        encoder = Encoder(layer)
        self.assertFalse(encoder.has_pos_enc)

    def test_has_merge_masks(self):
        self.assertTrue(hasattr(self.encode, 'merge_masks'))

    def test_merge_masks(self):
        self.assertTrue(callable(self.encode.merge_masks))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.encode, 'reset_parameters'))

    def test_reset_parameters_callable(self):
        self.assertTrue(callable(self.encode.reset_parameters))

    def test_call_reset_parameters(self):
        layer0 = patch.object(self.encode.layers[0], 'reset_parameters')
        pos_enc = patch.object(self.encode.pos_enc, 'reset_parameters')
        with layer0 as a, pos_enc as p:
            self.encode.reset_parameters()
            a.assert_called_once_with()
            p.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.encode, 'new'))

    def test_new_callable(self):
        self.assertTrue(callable(self.encode.new))

    def test_call_new(self):
        new = self.encode.new()
        self.assertIsInstance(new, Encoder)
        self.assertIsNot(new, self.encode)
        self.assertIsInstance(new.pos_enc, IdentityBlock)
        self.assertIsNot(new.pos_enc, self.encode.pos_enc)
        self.assertEqual(new.dropout, self.encode.dropout)
        self.assertEqual(new.device, self.encode.device)
        self.assertEqual(new.dtype, self.encode.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.n_layers = 3
        self.pad_id = 1
        self.dropout = 0.2
        self.dtype = pt.double
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            norm_cls=pt.nn.RMSNorm
        )
        self.encode = Encoder(
            self.layer,
            self.n_layers,
            self.pos_enc,
            self.dropout,
            dtype=self.dtype
        )

    def test_n_layers(self):
        self.assertEqual(self.n_layers, self.encode.n_layers)

    def test_pos_enc(self):
        self.assertIsInstance(self.encode.pos_enc, Sinusoidal)

    def test_dropout(self):
        self.assertEqual(self.dropout, self.encode.dropout)

    def test_dtype(self):
        self.assertIs(self.dtype, self.encode.dtype)

    def test_layers(self):
        for layer in self.encode.layers:
            self.assertIs(layer.dtype, self.dtype)

    def test_context(self):
        self.assertEqual(self.context, self.encode.context)

    def test_double_pos_enc_warns(self):
        layer = EncoderLayer(self.attention, self.feedforward, self.pos_enc)
        with self.assertWarns(UserWarning):
            _ = Encoder(layer, pos_enc=self.pos_enc)

    def test_no_pos_enc_warns(self):
        with self.assertWarns(UserWarning):
            _ = Encoder(self.layer)


class TestMergeMasks(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.encode = Encoder(self.layer, 1)
        self.inp = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.out = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.attn_mask = pt.nn.Transformer.generate_square_subsequent_mask(
            self.context,
            device='cpu'
        )
        self.src_mask = pt.zeros(self.context, device='cpu')
        self.src_mask[1] = float('-inf')
        self.src_mask[4] = float('-inf')
        self.src_mask[7] = float('-inf')

    def test_is_causal_no_attn_mask_no_src_mask(self):
        mask = self.encode.merge_masks(None, None, True)
        self.assertIsNone(mask)

    def test_is_causal_attn_mask_no_src_mask(self):
        mask = self.encode.merge_masks(self.attn_mask, None, True)
        self.assertIsNone(mask)

    def test_is_causal_no_attn_mask_src_mask(self):
        mask = self.encode.merge_masks(None, self.src_mask, True)
        self.assertIsNone(mask)

    def test_is_causal_attn_mask_src_mask(self):
        mask = self.encode.merge_masks(self.attn_mask, self.src_mask, True)
        self.assertIsNone(mask)

    def test_is_not_causal_no_attn_mask_no_src_mask(self):
        mask = self.encode.merge_masks(None, None, False)
        self.assertIsNone(mask)

    def test_is_not_causal_attn_mask_no_src_mask(self):
        mask = self.encode.merge_masks(self.attn_mask, None, False)
        self.assertIs(mask, self.attn_mask)

    def test_is_not_causal_no_attn_mask_src_mask_unbatched(self):
        mask = self.encode.merge_masks(None, self.src_mask, False)
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        mask = self.encode.merge_masks(None, src_mask, False)
        expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        mask = self.encode.merge_masks(None, src_mask, False)
        expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_attn_mask_src_mask_unbatched(self):
        mask = self.encode.merge_masks(self.attn_mask, self.src_mask, False)
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        mask = self.encode.merge_masks(self.attn_mask, src_mask, False)
        expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        mask = self.encode.merge_masks(self.attn_mask, src_mask, False)
        expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_2d_attn_mask_3d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask,
            self.src_mask.unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = expected.unsqueeze(0).unsqueeze(0) + self.attn_mask
        pt.testing.assert_close(mask, expected)

    def test_2d_attn_mask_4d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask,
            self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_3d_attn_mask_1d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask,
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_2d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask.unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_3d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask.unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask + expected.unsqueeze(0).unsqueeze(0)
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_4d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = (
            self.attn_mask.unsqueeze(0) +
            expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_1d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask,
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0).unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_2d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask.unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = (
            self.attn_mask.unsqueeze(0).unsqueeze(0) + expected)
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_3d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask.unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = (
            self.attn_mask.unsqueeze(0).unsqueeze(0) +
            expected.unsqueeze(0)
        )
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_4d_src_mask(self):
        mask = self.encode.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = (
            self.attn_mask.unsqueeze(0).unsqueeze(0) +
            expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
        pt.testing.assert_close(mask, expected)


class TestLayerCalledWithCorrectMask(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.encode = Encoder(self.layer, 1)
        self.inp = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.out = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.attn_mask = pt.nn.Transformer.generate_square_subsequent_mask(
            self.context,
            device='cpu'
        )
        self.src_mask = pt.zeros(self.context, device='cpu')
        self.src_mask[1] = float('-inf')
        self.src_mask[4] = float('-inf')
        self.src_mask[7] = float('-inf')

    def test_default(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertIsInstance(is_causal, bool)
            self.assertTrue(is_causal)

    def test_is_causal_no_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, None, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, None, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_no_attn_mask_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, self.src_mask, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_attn_mask_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, self.src_mask, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_not_causal_no_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, None, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertFalse(is_causal)

    def test_is_not_causal_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, None, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIs(mask, self.attn_mask)
            self.assertFalse(is_causal)

    def test_is_not_causal_no_attn_mask_src_mask_unbatched(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, self.src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        inp = self.inp.expand(3, -1, -1)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(inp, None, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_attn_mask_src_mask_unbatched(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, self.src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        inp = self.inp.expand(3, -1, -1)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(inp, self.attn_mask, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_2d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask,
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = expected.unsqueeze(0).unsqueeze(0) + self.attn_mask
            pt.testing.assert_close(mask, expected)

    def test_2d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask,
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_3d_attn_mask_1d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask,
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_2d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask + expected.unsqueeze(0).unsqueeze(0)
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0) +
                expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_1d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask,
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0).unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_2d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) + expected)
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) +
                expected.unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) +
                expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.attention = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward
        )
        self.encode = Encoder(self.layer, 2, self.pos_enc)
        self.inp = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.out = pt.rand(1, self.context, self.mod_dim, device='cpu')

    def test_pos_enc_called(self):
        with patch.object(
            self.encode.pos_enc,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.inp)

    def test_drop_called(self):
        with patch.object(
            self.encode.pos_enc,
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.drop,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out)

    def test_2nd_layer_called(self):
        with patch.object(
            self.encode.layers[0],
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.layers[1],
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out, None, True)

    def test_2d(self):
        inp = pt.rand(self.context, self.mod_dim, device='cpu')
        actual = self.encode(inp)
        expected = 1, self.context, self.mod_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_3d(self):
        inp = pt.rand(64, self.context, self.mod_dim, device='cpu')
        actual = self.encode(inp)
        expected = 64, self.context, self.mod_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_4d(self):
        inp = pt.rand(12, 64, self.context, self.mod_dim, device='cpu')
        actual = self.encode(inp)
        expected = 12, 64, self.context, self.mod_dim
        self.assertTupleEqual(expected, actual.shape)


if __name__ == '__main__':
    unittest.main()

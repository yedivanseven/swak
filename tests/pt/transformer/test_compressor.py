import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.blocks import ActivatedBlock, IdentityBlock
from swak.pt.transformer import (
    MultiheadedSelfAttention,
    EncoderLayer,
    Encoder,
    Sinusoidal,
    Compressor
)


def make_model(mod_dim, n_heads, context):
    layer = EncoderLayer(
        MultiheadedSelfAttention(mod_dim, n_heads),
        ActivatedBlock(mod_dim),
        pos_enc=Sinusoidal(mod_dim, context)
    )
    return Encoder(layer)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.attend = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=pos_enc
        )
        self.forward = ActivatedBlock(self.mod_dim)
        self.model = make_model(self.mod_dim, self.n_heads, self.context)
        self.compressor = Compressor(self.model, self.attend, self.forward)

    def test_has_model(self):
        self.assertTrue(hasattr(self.compressor, 'model'))

    def test_model(self):
        self.assertIs(self.compressor.model, self.model)

    def test_has_attend_inp(self):
        self.assertTrue(hasattr(self.compressor, 'attend_inp'))

    def test_attend_inp(self):
        self.assertIsInstance(
            self.compressor.attend_inp,
            MultiheadedSelfAttention
        )
        self.assertIsNot(self.compressor.attend_inp, self.attend)

    def test_has_attend_out(self):
        self.assertTrue(hasattr(self.compressor, 'attend_out'))

    def test_attend_out(self):
        self.assertIsInstance(
            self.compressor.attend_out,
            MultiheadedSelfAttention
        )
        self.assertIsNot(self.compressor.attend_out, self.attend)
        self.assertIsNot(
            self.compressor.attend_out,
            self.compressor.attend_inp
        )

    def test_has_forward_inp(self):
        self.assertTrue(hasattr(self.compressor, 'forward_inp'))

    def test_forward_inp(self):
        self.assertIsInstance(self.compressor.forward_inp, ActivatedBlock)
        self.assertIsNot(self.compressor.forward_inp, self.forward)

    def test_has_forward_out(self):
        self.assertTrue(hasattr(self.compressor, 'forward_out'))

    def test_forward_out(self):
        self.assertIsInstance(self.compressor.forward_out, ActivatedBlock)
        self.assertIsNot(self.compressor.forward_out, self.forward)
        self.assertIsNot(
            self.compressor.forward_out,
            self.compressor.forward_inp
        )

    def test_has_pos_enc(self):
        self.assertTrue(hasattr(self.compressor, 'pos_enc'))

    def test_pos_enc(self):
        self.assertIsInstance(self.compressor.pos_enc, IdentityBlock)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.compressor, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.compressor.bias, bool)
        self.assertTrue(self.compressor.bias)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.compressor, 'dropout'))

    def test_dropout(self):
        self.assertIsInstance(self.compressor.dropout, float)
        self.assertEqual(0.0, self.compressor.dropout)

    def test_has_norm_first(self):
        self.assertTrue(hasattr(self.compressor, 'norm_first'))

    def test_norm_first(self):
        self.assertIsInstance(self.compressor.norm_first, bool)
        self.assertTrue(self.compressor.norm_first)

    def test_has_norm_cls(self):
        self.assertTrue(hasattr(self.compressor, 'norm_cls'))

    def test_norm_cls(self):
        self.assertIs(self.compressor.norm_cls, pt.nn.LayerNorm)

    def test_has_args(self):
        self.assertTrue(hasattr(self.compressor, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.compressor.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.compressor, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.compressor.kwargs)

    def test_has_device(self):
        self.assertTrue(hasattr(self.compressor, 'device'))

    def test_device(self):
        self.assertIsInstance(self.compressor.device, pt.device)
        self.assertEqual('cpu', self.compressor.device.type)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.compressor, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.compressor.dtype, pt.float)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.compressor, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.compressor.drop, pt.nn.Dropout)
        self.assertEqual(self.compressor.dropout, self.compressor.drop.p)

    def test_has_compress(self):
        self.assertTrue(hasattr(self.compressor, 'compress'))

    def test_compress(self):
        self.assertIsInstance(
            self.compressor.compress,
            pt.nn.MultiheadAttention
        )
        self.assertEqual(self.mod_dim, self.compressor.compress.embed_dim)
        self.assertEqual(self.n_heads, self.compressor.compress.num_heads)
        self.assertTrue(self.compressor.compress.batch_first)
        self.assertEqual(
            self.compressor.device,
            self.compressor.compress.in_proj_weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.compress.in_proj_weight.dtype
        )

    def test_has_inflate(self):
        self.assertTrue(hasattr(self.compressor, 'inflate'))

    def test_inflate(self):
        self.assertIsInstance(
            self.compressor.inflate,
            pt.nn.MultiheadAttention
        )
        self.assertEqual(self.mod_dim, self.compressor.inflate.embed_dim)
        self.assertEqual(self.n_heads, self.compressor.inflate.num_heads)
        self.assertTrue(self.compressor.inflate.batch_first)
        self.assertEqual(
            self.compressor.device,
            self.compressor.inflate.in_proj_weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.inflate.in_proj_weight.dtype
        )

    def test_has_norm_self_attn_inp(self):
        self.assertTrue(hasattr(self.compressor, 'norm_self_attn_inp'))

    def test_norm_self_attn_inp(self):
        self.assertIsInstance(
            self.compressor.norm_self_attn_inp,
            pt.nn.LayerNorm
        )
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_self_attn_inp.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_self_attn_inp.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_self_attn_inp.weight.dtype
        )

    def test_has_norm_cross_attn_inp(self):
        self.assertTrue(hasattr(self.compressor, 'norm_cross_attn_inp'))

    def test_norm_cross_attn_inp(self):
        self.assertIsInstance(
            self.compressor.norm_cross_attn_inp, pt.nn.LayerNorm
        )
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_cross_attn_inp.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_cross_attn_inp.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_cross_attn_inp.weight.dtype
        )

    def test_has_norm_fwd_inp(self):
        self.assertTrue(hasattr(self.compressor, 'norm_fwd_inp'))

    def test_norm_fwd_inp(self):
        self.assertIsInstance(self.compressor.norm_fwd_inp, pt.nn.LayerNorm)
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_fwd_inp.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_fwd_inp.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_fwd_inp.weight.dtype
        )

    def test_has_norm_self_attn_out(self):
        self.assertTrue(hasattr(self.compressor, 'norm_self_attn_out'))

    def test_norm_self_attn_out(self):
        self.assertIsInstance(
            self.compressor.norm_self_attn_out, pt.nn.LayerNorm
        )
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_self_attn_out.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_self_attn_out.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_self_attn_out.weight.dtype
        )

    def test_has_norm_cross_attn_out(self):
        self.assertTrue(hasattr(self.compressor, 'norm_cross_attn_out'))

    def test_norm_cross_attn_out(self):
        self.assertIsInstance(
            self.compressor.norm_cross_attn_out, pt.nn.LayerNorm
        )
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_cross_attn_out.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_cross_attn_out.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_cross_attn_out.weight.dtype
        )

    def test_has_norm_fwd_out(self):
        self.assertTrue(hasattr(self.compressor, 'norm_fwd_out'))

    def test_norm_fwd_out(self):
        self.assertIsInstance(self.compressor.norm_fwd_out, pt.nn.LayerNorm)
        self.assertTupleEqual(
            (self.mod_dim,),
            self.compressor.norm_fwd_out.normalized_shape
        )
        self.assertEqual(
            self.compressor.device,
            self.compressor.norm_fwd_out.weight.device
        )
        self.assertIs(
            self.compressor.dtype,
            self.compressor.norm_fwd_out.weight.dtype
        )

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.compressor, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.compressor.mod_dim, int)
        self.assertEqual(self.mod_dim, self.compressor.mod_dim)

    def test_has_n_heads(self):
        self.assertTrue(hasattr(self.compressor, 'n_heads'))

    def test_n_heads(self):
        self.assertIsInstance(self.compressor.n_heads, int)
        self.assertEqual(self.n_heads, self.compressor.n_heads)

    def test_has_context(self):
        self.assertTrue(hasattr(self.compressor, 'context'))

    def test_context(self):
        self.assertIsInstance(self.compressor.context, int)
        self.assertEqual(self.context, self.compressor.context)

    def test_has_has_pos_enc(self):
        self.assertTrue(hasattr(self.compressor, 'has_pos_enc'))

    def test_has_pos_enc_correct(self):
        self.assertIsInstance(self.compressor.has_pos_enc, bool)
        self.assertTrue(self.compressor.has_pos_enc)

    def test_has_merge_masks(self):
        self.assertTrue(hasattr(self.compressor, 'merge_masks'))

    def test_merge_masks_callable(self):
        self.assertTrue(callable(self.compressor.merge_masks))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.compressor, 'reset_parameters'))

    def test_reset_parameters_callable(self):
        self.assertTrue(callable(self.compressor.reset_parameters))

    def test_call_reset_parameters(self):
        with patch.object(
            self.compressor.model, 'reset_parameters'
        ) as model, patch.object(
            self.compressor.attend_inp, 'reset_parameters'
        ) as ai, patch.object(
            self.compressor.attend_out, 'reset_parameters'
        ) as ao, patch.object(
            self.compressor.forward_inp, 'reset_parameters'
        ) as fi, patch.object(
            self.compressor.forward_out, 'reset_parameters'
        ) as fo, patch.object(
            self.compressor.pos_enc, 'reset_parameters'
        ) as pos,patch.object(
            self.compressor.compress, '_reset_parameters'
        ) as cmp, patch.object(
            self.compressor.inflate, '_reset_parameters'
        ) as inf, patch.object(
            self.compressor.norm_self_attn_inp, 'reset_parameters'
        ) as nai, patch.object(
            self.compressor.norm_self_attn_out, 'reset_parameters'
        ) as nao, patch.object(
            self.compressor.norm_cross_attn_inp, 'reset_parameters'
        ) as nci, patch.object(
            self.compressor.norm_cross_attn_out, 'reset_parameters'
        ) as nco, patch.object(
            self.compressor.norm_fwd_inp, 'reset_parameters'
        ) as nfi, patch.object(
            self.compressor.norm_fwd_out, 'reset_parameters'
        ) as nfo:
            self.compressor.reset_parameters()
            model.assert_called_once_with()
            ai.assert_called_once_with()
            ao.assert_called_once_with()
            fi.assert_called_once_with()
            fo.assert_called_once_with()
            pos.assert_called_once_with()
            cmp.assert_called_once_with()
            inf.assert_called_once_with()
            nai.assert_called_once_with()
            nao.assert_called_once_with()
            nci.assert_called_once_with()
            nco.assert_called_once_with()
            nfi.assert_called_once_with()
            nfo.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.compressor, 'new'))

    def test_new_callable(self):
        self.assertTrue(callable(self.compressor.new))

    def test_call_new(self):
        new = self.compressor.new()
        self.assertIsInstance(new, Compressor)
        self.assertIsNot(new, self.compressor)
        self.assertIsInstance(new.pos_enc, IdentityBlock)
        self.assertIsNot(new.pos_enc, self.compressor.pos_enc)
        self.assertEqual(new.bias, self.compressor.bias)
        self.assertEqual(new.dropout, self.compressor.dropout)
        self.assertEqual(new.norm_first, self.compressor.norm_first)
        self.assertIs(new.norm_cls, self.compressor.norm_cls)
        self.assertTupleEqual(new.args, self.compressor.args)
        self.assertEqual(new.device, self.compressor.device)
        self.assertIs(new.dtype, self.compressor.dtype)
        self.assertDictEqual(new.kwargs, self.compressor.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.dtype = pt.double
        self.dropout = 0.2
        self.attend = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        self.forward = ActivatedBlock(self.mod_dim)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.model = make_model(self.mod_dim, self.n_heads, self.context)
        self.compressor = Compressor(
            self.model,
            self.attend,
            self.forward,
            self.pos_enc,
            False,
            self.dropout,
            False,
            pt.nn.RMSNorm,
            1e-4,
            dtype=self.dtype,
            elementwise_affine=False
        )

    def test_pos_enc(self):
        self.assertIsInstance(self.compressor.pos_enc, Sinusoidal)

    def test_bias(self):
        self.assertFalse(self.compressor.bias)

    def test_dropout(self):
        self.assertEqual(self.dropout, self.compressor.dropout)

    def test_norm_first(self):
        self.assertFalse(self.compressor.norm_first)

    def test_norm_cls(self):
        self.assertIs(self.compressor.norm_cls, pt.nn.RMSNorm)

    def test_args(self):
        self.assertTupleEqual((1e-4,), self.compressor.args)

    def test_kwargs(self):
        self.assertDictEqual(
            {'elementwise_affine': False}, self.compressor.kwargs
        )

    def test_dtype(self):
        self.assertIs(self.dtype, self.compressor.dtype)

    def test_all_norms_are_rms(self):
        norms = [
            self.compressor.norm_self_attn_inp,
            self.compressor.norm_cross_attn_inp,
            self.compressor.norm_fwd_inp,
            self.compressor.norm_self_attn_out,
            self.compressor.norm_cross_attn_out,
            self.compressor.norm_fwd_out,
        ]
        for norm in norms:
            with self.subTest(norm=norm):
                self.assertIsInstance(norm, pt.nn.RMSNorm)
                self.assertAlmostEqual(1e-4, norm.eps)
                self.assertFalse(norm.elementwise_affine)

    def test_compress_no_bias(self):
        self.assertIsNone(self.compressor.compress.in_proj_bias)

    def test_inflate_no_bias(self):
        self.assertIsNone(self.compressor.inflate.in_proj_bias)

    def test_context(self):
        self.assertEqual(self.context, self.compressor.context)

    def test_has_pos_enc_correct(self):
        self.assertTrue(self.compressor.has_pos_enc)

    def test_call_new_with_pos_enc(self):
        new = self.compressor.new()
        self.assertIsInstance(new.pos_enc, Sinusoidal)
        self.assertIsNot(new.pos_enc, self.compressor.pos_enc)
        self.assertIs(new.dtype, self.compressor.dtype)

    def test_double_pos_enc_warns(self):
        attend = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            pos_enc=Sinusoidal(self.mod_dim, self.context)
        )
        with self.assertWarns(UserWarning):
            _ = Compressor(self.model, attend, self.forward, self.pos_enc)

    def test_no_pos_enc_warns(self):
        with self.assertWarns(UserWarning):
            _ = Compressor(self.model, self.attend, self.forward)


class TestMergeMasks(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        attend = MultiheadedSelfAttention(
            self.mod_dim, self.n_heads, pos_enc=pos_enc
        )
        forward = ActivatedBlock(self.mod_dim)
        self.compressor = Compressor(
            make_model(self.mod_dim, self.n_heads, self.context),
            attend,
            forward
        )
        self.attn_mask = pt.nn.Transformer.generate_square_subsequent_mask(
            self.context,
            device='cpu'
        )
        self.src_mask = pt.zeros(self.context, device='cpu')
        self.src_mask[1] = float('-inf')
        self.src_mask[4] = float('-inf')
        self.src_mask[7] = float('-inf')

    def test_is_causal_no_attn_mask_no_src_mask(self):
        mask = self.compressor.merge_masks(None, None, True)
        self.assertIsNone(mask)

    def test_is_causal_attn_mask_no_src_mask(self):
        mask = self.compressor.merge_masks(self.attn_mask, None, True)
        self.assertIsNone(mask)

    def test_is_causal_no_attn_mask_src_mask(self):
        mask = self.compressor.merge_masks(None, self.src_mask, True)
        self.assertIsNone(mask)

    def test_is_causal_attn_mask_src_mask(self):
        mask = self.compressor.merge_masks(self.attn_mask, self.src_mask, True)
        self.assertIsNone(mask)

    def test_is_not_causal_no_attn_mask_no_src_mask(self):
        mask = self.compressor.merge_masks(None, None, False)
        self.assertIsNone(mask)

    def test_is_not_causal_attn_mask_no_src_mask(self):
        mask = self.compressor.merge_masks(self.attn_mask, None, False)
        self.assertIs(mask, self.attn_mask)

    def test_is_not_causal_no_attn_mask_src_mask_unbatched(self):
        mask = self.compressor.merge_masks(None, self.src_mask, False)
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        mask = self.compressor.merge_masks(None, src_mask, False)
        expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        mask = self.compressor.merge_masks(None, src_mask, False)
        expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected)

    def test_is_not_causal_attn_mask_src_mask_unbatched(self):
        mask = self.compressor.merge_masks(
            self.attn_mask,
            self.src_mask,
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        mask = self.compressor.merge_masks(self.attn_mask, src_mask, False)
        expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        mask = self.compressor.merge_masks(self.attn_mask, src_mask, False)
        expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_2d_attn_mask_3d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask,
            self.src_mask.unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = expected.unsqueeze(0).unsqueeze(0) + self.attn_mask
        pt.testing.assert_close(mask, expected)

    def test_2d_attn_mask_4d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask,
            self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_3d_attn_mask_1d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask,
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_2d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask.unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_3d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask.unsqueeze(0),
            self.src_mask.unsqueeze(0).unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask + expected.unsqueeze(0).unsqueeze(0)
        pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_4d_src_mask(self):
        mask = self.compressor.merge_masks(
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
        mask = self.compressor.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask,
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = self.attn_mask.unsqueeze(0).unsqueeze(0) + expected
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_2d_src_mask(self):
        mask = self.compressor.merge_masks(
            self.attn_mask.unsqueeze(0).unsqueeze(0),
            self.src_mask.unsqueeze(0),
            False
        )
        expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
        expected = (
            self.attn_mask.unsqueeze(0).unsqueeze(0) + expected)
        pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_3d_src_mask(self):
        mask = self.compressor.merge_masks(
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
        mask = self.compressor.merge_masks(
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


class TestPad(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        attend = MultiheadedSelfAttention(
            self.mod_dim, self.n_heads, pos_enc=pos_enc
        )
        forward = ActivatedBlock(self.mod_dim)
        self.compressor = Compressor(
            make_model(self.mod_dim, self.n_heads, self.context),
            attend,
            forward
        )

    def test_even_no_mask_seq_unchanged(self):
        src = pt.rand(1, 4, self.mod_dim)
        padded, _ = self.compressor._pad(src, None)
        self.assertIs(padded, src)

    def test_even_no_mask_mask_unchanged(self):
        src = pt.rand(1, 4, self.mod_dim)
        _, pad_mask = self.compressor._pad(src, None)
        self.assertIsNone(pad_mask)

    def test_even_with_mask_seq_unchanged(self):
        src = pt.rand(1, 4, self.mod_dim)
        mask = pt.zeros(4, 4)
        padded, _ = self.compressor._pad(src, mask)
        self.assertIs(padded, src)

    def test_even_with_mask_mask_unchanged(self):
        src = pt.rand(1, 4, self.mod_dim)
        mask = pt.zeros(4, 4)
        _, pad_mask = self.compressor._pad(src, mask)
        self.assertIs(pad_mask, mask)

    def test_odd_no_mask_seq_shape(self):
        src = pt.rand(1, 5, self.mod_dim)
        padded, _ = self.compressor._pad(src, None)
        self.assertEqual(6, padded.size(-2))

    def test_odd_no_mask_first_token_prepended(self):
        src = pt.rand(1, 5, self.mod_dim)
        padded, _ = self.compressor._pad(src, None)
        pt.testing.assert_close(padded[..., 0, :], src[..., 0, :])
        pt.testing.assert_close(padded[..., 1:, :], src)

    def test_odd_no_mask_mask_is_none(self):
        src = pt.rand(1, 5, self.mod_dim)
        _, pad_mask = self.compressor._pad(src, None)
        self.assertIsNone(pad_mask)

    def test_odd_with_mask_seq_shape(self):
        src = pt.rand(1, 5, self.mod_dim)
        padded, _ = self.compressor._pad(src, pt.zeros(5, 5))
        self.assertTupleEqual((1, 6, self.mod_dim), tuple(padded.shape))

    def test_odd_with_mask_mask_shape(self):
        src = pt.rand(1, 5, self.mod_dim)
        _, pad_mask = self.compressor._pad(src, pt.zeros(5, 5))
        self.assertTupleEqual((6, 6), tuple(pad_mask.shape))

    def test_odd_with_mask_first_row_prepended(self):
        src = pt.rand(1, 5, self.mod_dim)
        mask = pt.rand(5, 5)
        _, pad_mask = self.compressor._pad(src, mask)
        pt.testing.assert_close(pad_mask[0], pad_mask[1])

    def test_odd_with_mask_first_col_prepended(self):
        src = pt.rand(1, 5, self.mod_dim)
        mask = pt.rand(5, 5)
        _, pad_mask = self.compressor._pad(src, mask)
        pt.testing.assert_close(pad_mask[:, 0], pad_mask[:, 1])


class TestShrink(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        attend = MultiheadedSelfAttention(
            self.mod_dim, self.n_heads, pos_enc=pos_enc
        )
        forward = ActivatedBlock(self.mod_dim)
        self.compressor = Compressor(
            make_model(self.mod_dim, self.n_heads, self.context),
            attend,
            forward
        )
        neginf = float('-inf')
        self.pad_len = 4
        # Non-trivial mask: top-left and bottom-right 2x2 blocks are blocked,
        # so all three shrunken masks carry meaningful -inf entries.
        self.pad_mask = pt.tensor([
            [neginf, neginf, neginf, 0.    ],
            [neginf, neginf, 0.,     0.    ],
            [0.,     0.,     neginf, neginf],
            [0.,     0.,     neginf, neginf],
        ])

    def test_not_causal_no_mask_inp_is_none(self):
        inp, _, _ = self.compressor._shrink(self.pad_len, None, False)
        self.assertIsNone(inp)

    def test_not_causal_no_mask_shrunk_is_none(self):
        _, shrunk, _ = self.compressor._shrink(self.pad_len, None, False)
        self.assertIsNone(shrunk)

    def test_not_causal_no_mask_out_is_none(self):
        _, _, out = self.compressor._shrink(self.pad_len, None, False)
        self.assertIsNone(out)

    def test_causal_no_mask_shrunk_is_none(self):
        _, shrunk, _ = self.compressor._shrink(self.pad_len, None, True)
        self.assertIsNone(shrunk)

    def test_causal_no_mask_inp_shape(self):
        inp, _, _ = self.compressor._shrink(self.pad_len, None, True)
        self.assertTupleEqual(
            (self.pad_len // 2, self.pad_len),
            tuple(inp.shape)
        )

    def test_causal_no_mask_inp_content(self):
        # Each compressed query j can attend to original keys 0..2j+1 only.
        inp, _, _ = self.compressor._shrink(self.pad_len, None, True)
        neginf = float('-inf')
        expected = pt.tensor([
            [0.,    0.,    neginf, neginf],
            [0.,    0.,    0.,     0.    ],
        ])
        pt.testing.assert_close(inp, expected)

    def test_causal_no_mask_out_shape(self):
        _, _, out = self.compressor._shrink(self.pad_len, None, True)
        self.assertTupleEqual(
            (self.pad_len, self.pad_len // 2),
            tuple(out.shape)
        )

    def test_causal_no_mask_out_content(self):
        # Original query i can only attend to compressed key j if 2j <= i.
        _, _, out = self.compressor._shrink(self.pad_len, None, True)
        neginf = float('-inf')
        expected = pt.tensor([
            [0.,    neginf],
            [0.,    neginf],
            [0.,    0.    ],
            [0.,    0.    ],
        ])
        pt.testing.assert_close(out, expected)

    def test_mask_provided_inp_shape(self):
        inp, _, _ = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        self.assertTupleEqual(
            (self.pad_len // 2, self.pad_len),
            tuple(inp.shape)
        )

    def test_mask_provided_inp_content(self):
        # inp[i, k] = max(pad_mask[2i, k], pad_mask[2i+1, k]):
        # a compressed query is blocked from key k only if both source
        # rows that contribute to it are blocked.
        inp, _, _ = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        neginf = float('-inf')
        expected = pt.tensor([
            [neginf, neginf, 0.,     0.    ],
            [0.,     0.,     neginf, neginf],
        ])
        pt.testing.assert_close(inp, expected)

    def test_mask_provided_shrunk_shape(self):
        _, shrunk, _ = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        self.assertTupleEqual(
            (self.pad_len // 2, self.pad_len // 2),
            tuple(shrunk.shape)
        )

    def test_mask_provided_shrunk_content(self):
        # shrunk[i, j] = max(inp[i, 2j], inp[i, 2j+1]):
        # compressed query i is blocked from compressed key j only if it is
        # blocked from both original keys that map to j.
        _, shrunk, _ = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        neginf = float('-inf')
        expected = pt.tensor([
            [neginf, 0.    ],
            [0.,     neginf],
        ])
        pt.testing.assert_close(shrunk, expected)

    def test_mask_provided_out_shape(self):
        _, _, out = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        self.assertTupleEqual(
            (self.pad_len, self.pad_len // 2),
            tuple(out.shape)
        )

    def test_mask_provided_out_content(self):
        # out[q, j] = max(pad_mask[q, 2j], pad_mask[q, 2j+1]):
        # original query q is blocked from compressed key j only if it is
        # blocked from both original keys that map to j.
        _, _, out = self.compressor._shrink(
            self.pad_len,
            self.pad_mask,
            False
        )
        neginf = float('-inf')
        expected = pt.tensor([
            [neginf, 0.    ],
            [neginf, 0.    ],
            [0.,     neginf],
            [0.,     neginf],
        ])
        pt.testing.assert_close(out, expected)


class TestUsageNormFirst(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        attend = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        forward = ActivatedBlock(self.mod_dim)
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.compressor = Compressor(
            make_model(self.mod_dim, self.n_heads, self.context),
            attend,
            forward,
            pos_enc
        )
        self.inp = pt.rand(1, self.context, self.mod_dim, device='cpu')

    def test_pos_enc_called(self):
        with patch.object(
            self.compressor.pos_enc,
            'forward',
            return_value=self.inp
        ) as pos_enc:
            _ = self.compressor(self.inp)
            pos_enc.assert_called_once()
            pt.testing.assert_close(pos_enc.call_args[0][0], self.inp)

    def test_norm_self_attn_inp_called_with_residual(self):
        # norm_first: norm is applied to residual before attend_inp
        with patch.object(
            self.compressor.pos_enc,
            'forward',
            return_value=self.inp
        ), patch.object(
            self.compressor.norm_self_attn_inp,
            'forward',
            return_value=self.inp
        ) as norm:
            _ = self.compressor(self.inp)
            norm.assert_called_once()
            pt.testing.assert_close(norm.call_args[0][0], self.inp)

    def test_attend_inp_receives_normed_input(self):
        # norm_first: attend_inp receives the normed residual
        with patch.object(
            self.compressor.norm_self_attn_inp,
            'forward',
            return_value=self.inp
        ), patch.object(
            self.compressor.attend_inp,
            'forward',
            return_value=self.inp
        ) as attend:
            _ = self.compressor(self.inp)
            attend.assert_called_once()
            pt.testing.assert_close(attend.call_args[0][0], self.inp)

    def test_attend_out_called(self):
        with patch.object(
            self.compressor.attend_out,
            'forward',
            return_value=self.inp
        ) as attend_out:
            _ = self.compressor(self.inp)
            attend_out.assert_called_once()

    def test_output_shape_2d(self):
        inp = pt.rand(self.context, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual(
            (1, self.context, self.mod_dim),
            tuple(out.shape)
        )

    def test_output_shape_3d_even(self):
        inp = pt.rand(8, self.context, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual(tuple(inp.shape), tuple(out.shape))

    def test_output_shape_3d_odd(self):
        inp = pt.rand(8, 7, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual((8, 7, self.mod_dim), tuple(out.shape))


class TestUsageNormLast(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        attend = MultiheadedSelfAttention(self.mod_dim, self.n_heads)
        forward = ActivatedBlock(self.mod_dim)
        pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.compressor = Compressor(
            make_model(self.mod_dim, self.n_heads, self.context),
            attend,
            forward,
            pos_enc,
            norm_first=False
        )
        self.inp = pt.rand(1, self.context, self.mod_dim, device='cpu')

    def test_attend_inp_receives_unnormed_input(self):
        # norm_last: attend_inp receives the raw pos_enc output, not normed
        with patch.object(
            self.compressor.pos_enc,
            'forward',
            return_value=self.inp
        ), patch.object(
            self.compressor.attend_inp,
            'forward',
            return_value=self.inp
        ) as attend:
            _ = self.compressor(self.inp)
            attend.assert_called_once()
            pt.testing.assert_close(attend.call_args[0][0], self.inp)

    def test_norm_self_attn_inp_receives_sum(self):
        # norm_last: norm receives (residual + drop(attended)),
        # with dropout=0 that equals pos_enc(src) + attend_inp(src) = 2 * inp
        with patch.object(
            self.compressor.pos_enc,
            'forward',
            return_value=self.inp
        ), patch.object(
            self.compressor.attend_inp,
            'forward',
            return_value=self.inp
        ), patch.object(
            self.compressor.norm_self_attn_inp,
            'forward',
            return_value=self.inp
        ) as norm:
            _ = self.compressor(self.inp)
            norm.assert_called_once()
            pt.testing.assert_close(norm.call_args[0][0], 2 * self.inp)

    def test_attend_out_called(self):
        with patch.object(
            self.compressor.attend_out,
            'forward',
            return_value=self.inp
        ) as attend_out:
            _ = self.compressor(self.inp)
            attend_out.assert_called_once()

    def test_output_shape_2d(self):
        inp = pt.rand(self.context, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual(
            (1, self.context, self.mod_dim),
            tuple(out.shape)
        )

    def test_output_shape_3d_even(self):
        inp = pt.rand(8, self.context, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual(tuple(inp.shape), tuple(out.shape))

    def test_output_shape_3d_odd(self):
        inp = pt.rand(8, 7, self.mod_dim)
        out = self.compressor(inp)
        self.assertTupleEqual((8, 7, self.mod_dim), tuple(out.shape))


if __name__ == '__main__':
    unittest.main()

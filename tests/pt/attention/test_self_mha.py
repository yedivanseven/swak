import sys
import unittest
from unittest.mock import patch, Mock
import torch as pt
from swak.pt.misc import BlockIdentity
from swak.pt.attention import MultiheadedSelfAttention


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.attention = MultiheadedSelfAttention(self.mod_dim)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.attention, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.attention.mod_dim, int)
        self.assertEqual(self.mod_dim, self.attention.mod_dim)

    def test_has_n_heads(self):
        self.assertTrue(hasattr(self.attention, 'n_heads'))

    def test_n_heads(self):
        self.assertIsInstance(self.attention.n_heads, int)
        self.assertEqual(1, self.attention.n_heads)

    def test_incompatible_n_heads_raises(self):
        with self.assertRaises(ValueError):
            _ = MultiheadedSelfAttention(16, 3)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.attention, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.attention.bias, bool)
        self.assertFalse(self.attention.bias)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.attention, 'dropout'))

    def test_dropout(self):
        self.assertEqual(0.1, self.attention.dropout)

    def test_has_pos_enc(self):
        self.assertTrue(hasattr(self.attention, 'pos_enc'))

    def test_pos_enc(self):
        self.assertIsInstance(self.attention.pos_enc, BlockIdentity)

    def test_pos_enc_to_called(self):
        pos_enc = Mock()
        pos_enc.to.return_value = pos_enc
        attention = MultiheadedSelfAttention(self.mod_dim, pos_enc=pos_enc)
        pos_enc.to.assert_called_once_with(
            device=attention.device.type,
            dtype=attention.dtype
        )

    def test_has_device(self):
        self.assertTrue(hasattr(self.attention, 'device'))

    def test_device(self):
        self.assertIsInstance(self.attention.device, pt.device)
        self.assertEqual('cpu', self.attention.device.type)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.attention, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.attention.dtype, pt.float)

    def test_has_qkv(self):
        self.assertTrue(hasattr(self.attention, 'qkv'))

    def test_qkv(self):
        self.assertIsInstance(self.attention.qkv, pt.nn.Linear)
        self.assertEqual(self.mod_dim, self.attention.qkv.in_features)
        self.assertEqual(3 * self.mod_dim, self.attention.qkv.out_features)
        self.assertEqual('cpu', self.attention.qkv.weight.device.type)
        self.assertIs(self.attention.qkv.weight.dtype, pt.float)
        self.assertIsNone(self.attention.qkv.bias)

    def test_has_out(self):
        self.assertTrue(hasattr(self.attention, 'out'))

    def test_out(self):
        self.assertIsInstance(self.attention.out, pt.nn.Linear)
        self.assertEqual(self.mod_dim, self.attention.out.in_features)
        self.assertEqual(self.mod_dim, self.attention.out.out_features)
        self.assertEqual('cpu', self.attention.out.weight.device.type)
        self.assertIs(self.attention.out.weight.dtype, pt.float)
        self.assertIsNone(self.attention.out.bias)

    def test_has_head_dim(self):
        self.assertTrue(hasattr(self.attention, 'head_dim'))

    def test_head_dim(self):
        self.assertIsInstance(self.attention.head_dim, int)
        self.assertEqual(self.mod_dim, self.attention.head_dim)

    def test_has_scale(self):
        self.assertTrue(hasattr(self.attention, 'scale'))

    def test_scale(self):
        expected = 1.0 / pt.sqrt(pt.tensor(self.mod_dim))
        self.assertEqual(expected, self.attention.scale)

    def test_has_has_pos_enc(self):
        self.assertTrue(hasattr(self.attention, 'has_pos_enc'))

    def test_has_pos_enc_correct(self):
        self.assertIsInstance(self.attention.has_pos_enc, bool)
        self.assertFalse(self.attention.has_pos_enc)

    def test_has_context(self):
        self.assertTrue(hasattr(self.attention, 'context'))

    def test_context(self):
        self.assertIsInstance(self.attention.context, int)
        self.assertEqual(sys.maxsize, self.attention.context)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.attention, 'reset_parameters'))

    def test_reset_parameters_callable(self):
        self.assertTrue(callable(self.attention.reset_parameters))

    @patch('torch.nn.Linear.reset_parameters')
    def test_call_reset_parameters(self, mock):
        pos_enc = Mock()
        pos_enc.to.return_value = pos_enc
        attention = MultiheadedSelfAttention(self.mod_dim, pos_enc=pos_enc)
        self.assertEqual(2, mock.call_count)
        attention.reset_parameters()
        self.assertEqual(4, mock.call_count)
        pos_enc.reset_parameters.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.attention, 'new'))

    def test_new_callable(self):
        self.assertTrue(callable(self.attention.new))

    def test_call_new(self):
        pos_enc = Mock()
        pos_enc.to.return_value = pos_enc
        attention = MultiheadedSelfAttention(self.mod_dim, pos_enc=pos_enc)
        new = attention.new()
        self.assertIsInstance(new, MultiheadedSelfAttention)
        self.assertIsNot(new, attention)
        self.assertEqual(attention.mod_dim, new.mod_dim)
        self.assertEqual(attention.n_heads, new.n_heads)
        self.assertEqual(attention.bias, new.bias)
        self.assertEqual(attention.dropout, new.dropout)
        self.assertEqual(attention.device, new.device)
        self.assertEqual(attention.dtype, new.dtype)
        self.assertIsInstance(new.pos_enc, Mock)
        self.assertIsNot(new.pos_enc, attention.pos_enc)
        pos_enc.new.assert_called_once_with()


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 32
        self.n_heads = 4
        self.bias = True
        self.dropout = 0.2
        self.dtype = pt.double
        self.context = 1234
        self.pos_enc = Mock()
        self.pos_enc.to.return_value = self.pos_enc
        self.pos_enc.context = self.context
        self.head_dim = self.mod_dim // self.n_heads
        self.attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.pos_enc,
            dtype=self.dtype
        )

    def test_bias(self):
        self.assertEqual(self.bias, self.attention.bias)

    def test_dropout(self):
        self.assertEqual(self.dropout, self.attention.dropout)

    def test_dtype(self):
        self.assertEqual(self.dtype, self.attention.dtype)

    def test_pos_enc(self):
        self.assertIs(self.attention.pos_enc, self.pos_enc)

    def test_qkv(self):
        self.assertTupleEqual((self.mod_dim,), self.attention.out.bias.shape)

    def test_out(self):
        self.assertTupleEqual((self.mod_dim,), self.attention.out.bias.shape)

    def test_head_dim(self):
        self.assertEqual(self.head_dim, self.attention.head_dim)

    def test_has_pos_enc_correct(self):
        self.assertTrue(self.attention.has_pos_enc)

    def test_context(self):
        self.assertEqual(self.context, self.attention.context)

    def test_call_new(self):
        new = self.attention.new()
        self.assertIsInstance(new, MultiheadedSelfAttention)
        self.assertEqual(self.attention.mod_dim, new.mod_dim)
        self.assertEqual(self.attention.n_heads, new.n_heads)
        self.assertEqual(self.attention.bias, new.bias)
        self.assertEqual(self.attention.dropout, new.dropout)
        self.assertEqual(self.attention.device, new.device)
        self.assertEqual(self.attention.dtype, new.dtype)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.batch_size = 15
        self.seq_len = 32
        self.mod_dim = 16
        self.n_heads = 2
        self.head_dim = self.mod_dim // self.n_heads
        self.attention = MultiheadedSelfAttention(
            self.mod_dim,
            self.n_heads,
            bias=False
        )
        self.attention.qkv.weight.data = pt.ones(
            3 * self.mod_dim,
            self.mod_dim,
            device='cpu'
        )
        self.attention.out.weight.data = pt.ones(
            self.mod_dim,
            self.mod_dim,
            device='cpu'
        )

    @patch('torch.nn.Linear.forward')
    def test_qkv_called(self, qkv):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        qkv.return_value = pt.rand(
            self.batch_size,
            self.seq_len,
            3 * self.mod_dim,
            device='cpu'
        )
        _ = self.attention(inp)
        self.assertEqual(2, qkv.call_count)
        pt.testing.assert_close(qkv.call_args_list[0][0][0], inp)

    @patch('torch.nn.functional.scaled_dot_product_attention')
    def test_attn_called_train_causal(self, attn):
        inp = pt.ones(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        out = pt.ones(
            15,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            device='cpu'
        )
        attn.return_value = out
        self.attention.train()
        _ = self.attention(inp, object(), True)
        pt.testing.assert_close(attn.call_args[1]['query'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['key'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['value'], 16 * out)
        self.assertIsNone(attn.call_args[1]['attn_mask'])
        self.assertEqual(
            self.attention.dropout,
            attn.call_args[1]['dropout_p']
        )
        self.assertIsInstance(attn.call_args[1]['is_causal'], bool)
        self.assertTrue(attn.call_args[1]['is_causal'])
        self.assertEqual(self.attention.scale, attn.call_args[1]['scale'])

    @patch('torch.nn.functional.scaled_dot_product_attention')
    def test_attn_called_eval_causal(self, attn):
        inp = pt.ones(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        out = pt.ones(
            15,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            device='cpu'
        )
        attn.return_value = out
        self.attention.eval()
        _ = self.attention(inp, object(), True)
        pt.testing.assert_close(attn.call_args[1]['query'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['key'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['value'], 16 * out)
        self.assertIsNone(attn.call_args[1]['attn_mask'])
        self.assertEqual(0.0,  attn.call_args[1]['dropout_p'])
        self.assertIsInstance(attn.call_args[1]['is_causal'], bool)
        self.assertTrue(attn.call_args[1]['is_causal'])
        self.assertEqual(self.attention.scale, attn.call_args[1]['scale'])

    @patch('torch.nn.functional.scaled_dot_product_attention')
    def test_attn_called_train_not_causal(self, attn):
        inp = pt.ones(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        out = pt.ones(
            15,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            device='cpu'
        )
        attn.return_value = out
        mask = object()
        self.attention.train()
        _ = self.attention(inp, mask, False)
        pt.testing.assert_close(attn.call_args[1]['query'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['key'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['value'], 16 * out)
        self.assertIs(attn.call_args[1]['attn_mask'], mask)
        self.assertEqual(
            self.attention.dropout,
            attn.call_args[1]['dropout_p']
        )
        self.assertIsInstance(attn.call_args[1]['is_causal'], bool)
        self.assertFalse(attn.call_args[1]['is_causal'])
        self.assertEqual(self.attention.scale, attn.call_args[1]['scale'])

    @patch('torch.nn.functional.scaled_dot_product_attention')
    def test_attn_called_eval_not_causal(self, attn):
        inp = pt.ones(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        out = pt.ones(
            15,
            self.n_heads,
            self.seq_len,
            self.head_dim,
            device='cpu'
        )
        attn.return_value = out
        mask = object()
        self.attention.eval()
        _ = self.attention(inp, mask, False)
        pt.testing.assert_close(attn.call_args[1]['query'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['key'], 16 * out)
        pt.testing.assert_close(attn.call_args[1]['value'], 16 * out)
        self.assertIs(attn.call_args[1]['attn_mask'], mask)
        self.assertEqual(0.0, attn.call_args[1]['dropout_p'])
        self.assertIsInstance(attn.call_args[1]['is_causal'], bool)
        self.assertFalse(attn.call_args[1]['is_causal'])
        self.assertEqual(self.attention.scale, attn.call_args[1]['scale'])

    @patch('torch.nn.Linear.forward')
    def test_out_called(self, out):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        out.return_value = pt.rand(
            self.batch_size,
            self.seq_len,
            3 * self.mod_dim,
            device='cpu'
        )
        _ = self.attention(inp)
        self.assertEqual(2, out.call_count)
        self.assertTupleEqual(
            (self.batch_size, self.seq_len, self.mod_dim),
            out.call_args[0][0].shape
        )

    def test_2d_inp_1d_mask(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_2d_inp_2d_mask(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(self.seq_len, self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_2d_inp_3d_mask_first_dim_1(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(1, self.seq_len, self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_2d_inp_3d_mask_first_dim_n_heads(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(self.n_heads, self.seq_len, self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_2d_inp_3d_mask_first_dim_wrong_raises(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(
            self.batch_size,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        with self.assertRaises(RuntimeError):
            _ = self.attention(inp, mask, False)

    def test_2d_inp_4d_mask_raises(self):
        inp = pt.rand(self.seq_len, self.mod_dim, device='cpu')
        mask = pt.zeros(1, 1, self.seq_len, self.seq_len, device='cpu')
        with self.assertRaises(RuntimeError):
            _ = self.attention(inp, mask, False)

    def test_3d_inp_1d_mask(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)


    def test_3d_inp_2d_mask(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(self.seq_len, self.seq_len, device='cpu')
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_3d_mask_batch_first_dim_1(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            1,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_3d_mask_first_dim_batch_size_raises(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.batch_size,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        with self.assertRaises(RuntimeError):
            _ = self.attention(inp, mask, False)

    def test_3d_inp_3d_mask_first_dim_n_heads(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.n_heads,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_3d_mask_n_heads(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.n_heads,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_4d_mask_first_dims_1(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            1,
            1,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_4d_mask_first_dim_1_second_dim_n_head(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            1,
            self.n_heads,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_4d_mask_first_dim_batch_size_second_dim_1(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.batch_size,
            1,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_4d_mask_first_dim_batch_size_second_dim_n_heads(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.batch_size,
            1,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        actual = self.attention(inp, mask, False)
        self.assertTupleEqual(inp.shape, actual.shape)

    def test_3d_inp_4d_mask_first_dims_wrong_raises(self):
        inp = pt.rand(
            self.batch_size,
            self.seq_len,
            self.mod_dim,
            device='cpu'
        )
        mask = pt.zeros(
            self.n_heads,
            self.batch_size,
            self.seq_len,
            self.seq_len,
            device='cpu'
        )
        with self.assertRaises(RuntimeError):
            _ = self.attention(inp, mask, False)


if __name__ == '__main__':
    unittest.main()

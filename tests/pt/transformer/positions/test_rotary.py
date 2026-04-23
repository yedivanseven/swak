import unittest
import torch as pt
from swak.pt.transformer import Rotary


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.context = 32
        self.n_heads = 2
        self.head_dim = self.mod_dim // self.n_heads
        self.pos_enc = Rotary(self.mod_dim, self.context, self.n_heads)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.pos_enc, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.pos_enc.mod_dim, int)
        self.assertEqual(self.mod_dim, self.pos_enc.mod_dim)

    def test_has_context(self):
        self.assertTrue(hasattr(self.pos_enc, 'context'))

    def test_context(self):
        self.assertIsInstance(self.pos_enc.context, int)
        self.assertEqual(self.context, self.pos_enc.context)

    def test_has_n_heads(self):
        self.assertTrue(hasattr(self.pos_enc, 'n_heads'))

    def test_n_heads(self):
        self.assertIsInstance(self.pos_enc.n_heads, int)
        self.assertEqual(self.n_heads, self.pos_enc.n_heads)

    def test_has_device(self):
        self.assertTrue(hasattr(self.pos_enc, 'device'))

    def test_device(self):
        self.assertIsInstance(self.pos_enc.device, pt.device)
        self.assertEqual('cpu', self.pos_enc.device.type)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.pos_enc, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.pos_enc.dtype, pt.float)

    def test_has_positional_encodings(self):
        self.assertTrue(hasattr(self.pos_enc, 'positional_encodings'))

    def test_positional_encodings_type(self):
        self.assertIsInstance(
            self.pos_enc.positional_encodings,
            pt.Tensor
        )

    def test_positional_encodings_device(self):
        self.assertEqual(
            self.pos_enc.device.type,
            self.pos_enc.positional_encodings.device.type
        )

    def test_positional_encodings_dtype(self):
        self.assertIs(
            self.pos_enc.positional_encodings.dtype,
            self.pos_enc.dtype
        )

    def test_positional_encodings_shape(self):
        self.assertTupleEqual(
            (1, self.context, self.head_dim // 2, 2),
            self.pos_enc.positional_encodings.data.shape
        )

    def test_positional_encodings_values(self):
        mod_dim = 8
        context = 3
        n_heads = 2
        head_dim = mod_dim // n_heads
        pos_enc = Rotary(mod_dim, context, n_heads)

        inp = pt.rand(n_heads, context, head_dim, device='cpu')


        theta1 = pt.tensor(context**(-0.0 / head_dim), device='cpu')
        theta2 = pt.tensor(context**(-2.0 / head_dim), device='cpu')

        rot_mat = pt.tensor([
            [
                [pt.cos(0 * theta1), -pt.sin(0 * theta1), 0.0, 0.0],
                [pt.sin(0 * theta1), +pt.cos(0 * theta1), 0.0, 0.0],
                [0.0, 0.0, pt.cos(0 * theta2), -pt.sin(0 * theta2)],
                [0.0, 0.0, pt.sin(0 * theta2), +pt.cos(0 * theta2)]
            ],
            [
                [pt.cos(1 * theta1), -pt.sin(1 * theta1), 0.0, 0.0],
                [pt.sin(1 * theta1), +pt.cos(1 * theta1), 0.0, 0.0],
                [0.0, 0.0, pt.cos(1 * theta2), -pt.sin(1 * theta2)],
                [0.0, 0.0, pt.sin(1 * theta2), +pt.cos(1 * theta2)]
            ],
            [
                [pt.cos(2 * theta1), -pt.sin(2 * theta1), 0.0, 0.0],
                [pt.sin(2 * theta1), +pt.cos(2 * theta1), 0.0, 0.0],
                [0.0, 0.0, pt.cos(2 * theta2), -pt.sin(2 * theta2)],
                [0.0, 0.0, pt.sin(2 * theta2), +pt.cos(2 * theta2)]
            ]
        ], device='cpu')

        actual = pos_enc(inp)
        pt.testing.assert_close(actual[0, 0], rot_mat[0] @ inp[0, 0])
        pt.testing.assert_close(actual[1, 0], rot_mat[0] @ inp[1, 0])
        pt.testing.assert_close(actual[0, 1], rot_mat[1] @ inp[0, 1])
        pt.testing.assert_close(actual[1, 1], rot_mat[1] @ inp[1, 1])
        pt.testing.assert_close(actual[0, 2], rot_mat[2] @ inp[0, 2])
        pt.testing.assert_close(actual[1, 2], rot_mat[2] @ inp[1, 2])

    def test_has_head_dim(self):
        self.assertTrue(hasattr(self.pos_enc, 'head_dim'))

    def test_head_dim(self):
        self.assertIsInstance(self.pos_enc.head_dim, int)
        self.assertEqual(self.head_dim, self.pos_enc.head_dim)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.pos_enc, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.pos_enc.reset_parameters))

    def test_call_reset_parameters(self):
        self.pos_enc.reset_parameters()

    def test_has_new(self):
        self.assertTrue(hasattr(self.pos_enc, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.pos_enc.new))

    def test_call_new(self):
        new = self.pos_enc.new()
        self.assertIsInstance(new, Rotary)
        self.assertIsNot(new, self.pos_enc)
        self.assertEqual(self.pos_enc.mod_dim, new.mod_dim)
        self.assertEqual(self.pos_enc.context, new.context)
        self.assertEqual(self.pos_enc.n_heads, new.n_heads)
        self.assertEqual(self.pos_enc.device.type, new.device.type)
        self.assertIs(self.pos_enc.dtype, new.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.context = 32
        self.n_heads = 2
        self.dtype = pt.double
        self.pos_enc = Rotary(
            self.mod_dim,
            self.context,
            self.n_heads,
            dtype=self.dtype
        )

    def test_dtype(self):
        self.assertIs(self.pos_enc.dtype, self.dtype)
        self.assertIs(self.pos_enc.positional_encodings.dtype, self.dtype)

    def test_raises_if_n_heads_does_not_divide_mod_dim(self):
        with self.assertRaises(ValueError):
            _ = Rotary(16, self.context, 3)

    def test_raised_if_head_dim_not_even(self):
        with self.assertRaises(ValueError):
            _ = Rotary(15, self.context, 3)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.context = 32
        self.n_heads = 2
        self.head_dim = self.mod_dim // self.n_heads
        self.pos_enc = Rotary(self.mod_dim, self.context, self.n_heads)

    def test_3d(self):
        inp = pt.zeros(
            self.n_heads,
            self.context,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = self.n_heads, self.context, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_3d_short(self):
        inp = pt.zeros(
            self.n_heads,
            self.context - 5,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = self.n_heads, self.context - 5, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_3d_offset(self):
        inp = pt.zeros(
            self.n_heads,
            self.context - 5,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp, 3)
        expected = self.n_heads, self.context - 5, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_4d(self):
        inp = pt.zeros(
            3,
            self.n_heads,
            self.context,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = 3, self.n_heads, self.context, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_4d_short(self):
        inp = pt.zeros(
            3,
            self.n_heads,
            self.context - 6,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = 3, self.n_heads, self.context - 6, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_4d_offset(self):
        inp = pt.zeros(
            3,
            self.n_heads,
            self.context - 6,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp, 4)
        expected = 3, self.n_heads, self.context - 6, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_5d(self):
        inp = pt.zeros(
            5,
            3,
            self.n_heads,
            self.context,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = 5, 3, self.n_heads, self.context, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_5d_short(self):
        inp = pt.zeros(
            5,
            3,
            self.n_heads,
            self.context - 6,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp)
        expected = 5, 3, self.n_heads, self.context - 6, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_5d_offset(self):
        inp = pt.zeros(
            5,
            3,
            self.n_heads,
            self.context - 6,
            self.head_dim,
            device='cpu'
        )
        actual = self.pos_enc(inp, 2)
        expected = 5, 3, self.n_heads, self.context - 6, self.head_dim
        self.assertTupleEqual(expected, actual.shape)

    def test_too_long_raises(self):
        inp = pt.zeros(
            3,
            self.n_heads,
            self.context + 1,
            self.head_dim,
            device='cpu'
        )
        with self.assertRaises(RuntimeError):
            _ = self.pos_enc(inp)


if __name__ == '__main__':
    unittest.main()

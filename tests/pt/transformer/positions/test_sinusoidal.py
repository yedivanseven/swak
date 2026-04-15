import unittest
import torch as pt
from swak.pt.transformer import Sinusoidal


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 8
        self.context = 16
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)

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
            (1, self.context, self.mod_dim),
            self.pos_enc.positional_encodings.data.shape
        )

    def test_positional_encodings_values(self):
        exponents = -pt.arange(0, self.mod_dim, 2) / self.mod_dim
        divisors = pt.tensor(self.context).pow(exponents)
        positions = pt.arange(0, self.context).unsqueeze(1)
        angles = positions * divisors
        expected = pt.empty(1, self.context, self.mod_dim, device='cpu')
        expected[:, :, 0::2] = pt.sin(angles)
        expected[:, :, 1::2] = pt.cos(angles)
        pt.testing.assert_close(self.pos_enc.positional_encodings, expected)

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
        self.assertIsInstance(new, Sinusoidal)
        self.assertIsNot(new, self.pos_enc)
        self.assertEqual(self.pos_enc.mod_dim, new.mod_dim)
        self.assertEqual(self.pos_enc.context, new.context)
        self.assertEqual(self.pos_enc.device.type, new.device.type)
        self.assertIs(self.pos_enc.dtype, new.dtype)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 8
        self.context = 16
        self.dtype = pt.double
        self.pos_enc = Sinusoidal(self.mod_dim, self.context, dtype=self.dtype)

    def test_dtype(self):
        self.assertIs(self.pos_enc.dtype, self.dtype)
        self.assertIs(self.pos_enc.positional_encodings.dtype, self.dtype)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 8
        self.context = 16
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)


    def test_2d(self):
        inp = pt.zeros(self.context, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(actual, self.pos_enc.positional_encodings)

    def test_2d_short(self):
        inp = pt.zeros(self.context - 4, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(
            actual,
            self.pos_enc.positional_encodings[:, :-4, :]
        )

    def test_3d(self):
        inp = pt.zeros(1, self.context, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(actual, self.pos_enc.positional_encodings)

    def test_3d_short(self):
        inp = pt.zeros(1, self.context - 5, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(
            actual,
            self.pos_enc.positional_encodings[:, :-5, :]
        )

    def test_4d(self):
        inp = pt.zeros(3, 1, self.context, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(actual[0], self.pos_enc.positional_encodings)

    def test_4d_short(self):
        inp = pt.zeros(3, 1, self.context - 6, self.mod_dim, device='cpu')
        actual = self.pos_enc(inp)
        pt.testing.assert_close(
            actual[0],
            self.pos_enc.positional_encodings[:, :-6, :]
        )

    def test_too_long_raises(self):
        inp = pt.zeros(3, 1, self.context + 1, self.mod_dim, device='cpu')
        with self.assertRaises(RuntimeError):
            _ = self.pos_enc(inp)


if __name__ == '__main__':
    unittest.main()

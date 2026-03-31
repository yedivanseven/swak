import unittest
from unittest.mock import Mock
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import Repeat, SkipConnection, ActivatedBlock


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.block = ActivatedBlock(4)
        self.skip = SkipConnection(self.block)
        self.repeat = Repeat(self.skip)

    def test_has_n_layers(self):
        self.assertTrue(hasattr(self.repeat, 'n_layers'))

    def test_n_layers(self):
        self.assertIsInstance(self.repeat.n_layers, int)
        self.assertEqual(2, self.repeat.n_layers)

    def test_has_blocks(self):
        self.assertTrue(hasattr(self.repeat, 'blocks'))

    def test_blocks(self):
        self.assertIsInstance(self.repeat.blocks, ptn.Sequential)
        self.assertEqual(2, len(self.repeat.blocks))

    def test_blocks_are_skip_connections(self):
        for block in self.repeat.blocks:
            self.assertIsInstance(block, SkipConnection)

    def test_blocks_are_distinct(self):
        self.assertIsNot(self.repeat.blocks[0], self.repeat.blocks[1])
        self.assertIsNot(self.repeat.blocks[0], self.block)
        self.assertIsNot(self.repeat.blocks[1], self.block)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.repeat, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.repeat.mod_dim, int)
        self.assertEqual(4, self.repeat.mod_dim)

    def test_has_device(self):
        self.assertTrue(hasattr(self.repeat, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.repeat.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.repeat, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.repeat.dtype, pt.float)
        self.assertIs(self.skip.dtype, self.repeat.dtype)

    def test_has_layers(self):
        self.assertTrue(hasattr(self.repeat, 'layers'))

    def test_layers(self):
        self.assertIsInstance(self.repeat.layers, range)
        self.assertTupleEqual((0, 1), tuple(self.repeat.layers))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.repeat, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.repeat.reset_parameters))

    def test_call_reset_parameters(self):
        for skip in self.repeat.blocks:
            skip.reset_parameters = Mock(return_value=skip)
        self.repeat.reset_parameters()
        for skip in self.repeat.blocks:
            skip.reset_parameters.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.repeat, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.repeat.new))

    def test_call_new(self):
        new = self.repeat.new()
        self.assertIsInstance(new, Repeat)
        self.assertEqual(self.repeat.n_layers, new.n_layers)
        self.assertEqual(self.skip.mod_dim, new.mod_dim)
        self.assertEqual(self.skip.device, new.device)
        self.assertIs(self.skip.dtype, new.dtype)

    def test_callable(self):
        self.assertTrue(callable(self.repeat))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.block = ActivatedBlock(4)
        self.skip = SkipConnection(self.block)
        self.repeat = Repeat(self.skip, 4, dtype=pt.float64)

    def test_n_layers(self):
        self.assertEqual(4, self.repeat.n_layers)

    def test_blocks(self):
        self.assertEqual(4, len(self.repeat.blocks))

    def test_dtype(self):
        self.assertIs(self.repeat.dtype, pt.float64)
        for block in self.repeat.blocks:
            self.assertIs(block.dtype, pt.float64)


class TestNLayers(unittest.TestCase):

    def setUp(self):
        self.block = ActivatedBlock(4)
        self.skip = SkipConnection(self.block)

    def test_n_layers_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            Repeat(self.skip, n_layers=0)

    def test_n_layers_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            Repeat(self.skip, n_layers=-1)

    def test_n_layers_non_castable_raises_type_error(self):
        with self.assertRaises(TypeError):
            Repeat(self.skip, n_layers='abc')

    def test_n_layers_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            Repeat(self.skip, n_layers=None)

    def test_n_layers_float_is_cast(self):
        repeat = Repeat(self.skip, n_layers=3.9)
        self.assertEqual(3, repeat.n_layers)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.block = ActivatedBlock(4)
        self.skip = SkipConnection(self.block)
        self.repeat = Repeat(self.skip, n_layers=3)

    def test_all_blocks_called(self):
        inp = pt.ones(4)
        mocks = []
        prev_out = inp
        for i, block in enumerate(self.repeat.blocks):
            out = pt.ones(4) * float(i + 2)
            m = Mock(return_value=out)
            block.forward = m
            mocks.append((m, prev_out))
            prev_out = out
        self.repeat(inp)
        for m, expected_inp in mocks:
            m.assert_called_once()
            pt.testing.assert_close(m.call_args[0][0], expected_inp)


if __name__ == '__main__':
    unittest.main()

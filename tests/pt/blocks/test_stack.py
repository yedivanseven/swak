import unittest
from unittest.mock import patch
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import Stack
from swak.pt.misc import Identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.stack = Stack(Identity())

    def test_has_skip(self):
        self.assertTrue(hasattr(self.stack, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.stack.skip, Identity)

    def test_has_n_layers(self):
        self.assertTrue(hasattr(self.stack, 'n_layers'))

    def test_n_layers(self):
        self.assertIsInstance(self.stack.n_layers, int)
        self.assertEqual(2, self.stack.n_layers)

    def test_has_sequence(self):
        self.assertTrue(hasattr(self.stack, 'sequence'))

    def test_sequence(self):
        self.assertIsInstance(self.stack.sequence, ptn.Sequential)
        self.assertEqual(2, len(self.stack.sequence))

    def test_has_layers(self):
        self.assertTrue(hasattr(self.stack, 'layers'))

    def test_layers(self):
        self.assertIsInstance(self.stack.layers, range)
        self.assertTupleEqual((0, 1), tuple(self.stack.layers))

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.stack, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.stack.reset_parameters))

    def test_call_reset_parameters(self):
        with patch.object(Identity, 'reset_parameters') as mock:
            self.stack.reset_parameters()
            self.assertEqual(2, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.stack, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.stack.new))

    def test_call_new(self):
        new = self.stack.new()
        self.assertIsInstance(new, Stack)
        self.assertIs(new.skip, self.stack.skip)
        self.assertEqual(self.stack.n_layers, new.n_layers)

    def test_callable(self):
        self.assertTrue(callable(self.stack))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.stack = Stack(
            Identity(),
            4
        )

    def test_n_layers(self):
        self.assertEqual(4, self.stack.n_layers)

    def test_sequence(self):
        self.assertEqual(4, len(self.stack.sequence))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.stack = Stack(Identity(), 3)

    def test_skip_called(self):
        with patch.object(Identity, 'forward') as mock:
            inp = pt.tensor(4)
            mock.return_value = inp
            _ = self.stack(inp)
            self.assertEqual(3, mock.call_count)
            arg1 = mock.call_args_list[0][0][0]
            arg2 = mock.call_args_list[1][0][0]
            arg3 = mock.call_args_list[2][0][0]
            self.assertIs(arg1, inp)
            self.assertIs(arg2, inp)
            self.assertIs(arg3, inp)


if __name__ == '__main__':
    unittest.main()

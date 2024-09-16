import unittest
from unittest.mock import patch
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import Repeat
from swak.pt.misc import Identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.repeat = Repeat(Identity())

    def test_has_skip(self):
        self.assertTrue(hasattr(self.repeat, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.repeat.skip, Identity)

    def test_has_n_layers(self):
        self.assertTrue(hasattr(self.repeat, 'n_layers'))

    def test_n_layers(self):
        self.assertIsInstance(self.repeat.n_layers, int)
        self.assertEqual(2, self.repeat.n_layers)

    def test_has_sequence(self):
        self.assertTrue(hasattr(self.repeat, 'sequence'))

    def test_sequence(self):
        self.assertIsInstance(self.repeat.sequence, ptn.Sequential)
        self.assertEqual(2, len(self.repeat.sequence))

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
        with patch.object(Identity, 'reset_parameters') as mock:
            self.repeat.reset_parameters()
            self.assertEqual(2, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.repeat, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.repeat.new))

    def test_call_new(self):
        new = self.repeat.new()
        self.assertIsInstance(new, Repeat)
        self.assertIs(new.skip, self.repeat.skip)
        self.assertEqual(self.repeat.n_layers, new.n_layers)

    def test_callable(self):
        self.assertTrue(callable(self.repeat))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.repeat = Repeat(
            Identity(),
            4
        )

    def test_n_layers(self):
        self.assertEqual(4, self.repeat.n_layers)

    def test_sequence(self):
        self.assertEqual(4, len(self.repeat.sequence))


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.repeat = Repeat(Identity(), 3)

    def test_skip_called(self):
        with patch.object(Identity, 'forward') as mock:
            inp = pt.tensor(4)
            mock.return_value = inp
            _ = self.repeat(inp)
            self.assertEqual(3, mock.call_count)
            arg1 = mock.call_args_list[0][0][0]
            arg2 = mock.call_args_list[1][0][0]
            arg3 = mock.call_args_list[2][0][0]
            self.assertIs(arg1, inp)
            self.assertIs(arg2, inp)
            self.assertIs(arg3, inp)

    def test_no_repetitions(self):
        repeat = Repeat(Identity(), 0)
        inp = pt.tensor(4)
        actual = repeat(inp)
        self.assertIs(actual, inp)


if __name__ == '__main__':
    unittest.main()

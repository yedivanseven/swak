import unittest
from unittest.mock import patch, Mock
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import Repeat, SkipConnection
from swak.pt.misc import Identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.block = Mock()
        self.skip = SkipConnection(self.block)
        self.repeat = Repeat(self.skip)

    def test_has_skip(self):
        self.assertTrue(hasattr(self.repeat, 'skip'))

    def test_skip(self):
        self.assertIsInstance(self.repeat.skip, SkipConnection)

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

    def test_has_norm(self):
        self.assertTrue(hasattr(self.repeat, 'norm'))

    def test_norm(self):
        self.assertIsInstance(self.repeat.norm, Identity)

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
            self.assertEqual(3, mock.call_count)

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
        self.block = Mock()
        self.skip = SkipConnection(self.block)
        self.repeat = Repeat(self.skip, 4)

    def test_n_layers(self):
        self.assertEqual(4, self.repeat.n_layers)

    def test_blocks(self):
        self.assertEqual(4, len(self.repeat.blocks))

    def test_norm_first(self):
        block = Mock()
        norm = Mock()
        norm_cls = Mock(return_value=norm)
        skip = SkipConnection(block, norm_cls=norm_cls, hello='world')
        repeat = Repeat(skip, 4)
        norm_cls.assert_called_with(hello='world')
        self.assertEqual(6, norm_cls.call_count)
        self.assertIs(repeat.norm, norm)

    def test_norm_after(self):
        block = Mock()
        norm = Mock()
        norm_cls = Mock(return_value=norm)
        skip = SkipConnection(
            block,
            norm_first=False,
            norm_cls=norm_cls,
            hello='world'
        )
        repeat = Repeat(skip, 4)
        norm_cls.assert_called_with(hello='world')
        self.assertEqual(5, norm_cls.call_count)
        self.assertIsInstance(repeat.norm, Identity)


class TestUsage(unittest.TestCase):

    def test_skip_called_norm_first(self):
        norm_out = pt.ones(4) * 2.0
        norm = Mock(return_value=norm_out)
        skip = Mock()
        skip.new = Mock(return_value=Identity())
        skip.norm_cls = Mock(return_value=norm)
        skip.norm_first = True
        skip.args = ()
        skip.kwargs = {}
        repeat = Repeat(skip, 3)
        inp = pt.ones(4)
        actual = repeat(inp)
        self.assertIs(actual, norm_out)
        norm.assert_called_once_with(inp)

    def test_skip_called_norm_after(self):
        norm_out = pt.ones(4) * 2.0
        norm = Mock(return_value=norm_out)
        skip = Mock()
        skip.new = Mock(return_value=Identity())
        skip.norm_cls = Mock(return_value=norm)
        skip.norm_first = False
        skip.args = ()
        skip.kwargs = {}
        repeat = Repeat(skip, 3)
        inp = pt.ones(4)
        actual = repeat(inp)
        self.assertIs(actual, inp)
        norm.assert_not_called()

    def test_no_repetitions_norm_first(self):
        block = Mock()
        expected = pt.ones(4) * 2
        norm = Mock(return_value=expected)
        norm_cls = Mock(return_value=norm)
        skip = SkipConnection(block, norm_cls=norm_cls)
        repeat = Repeat(skip, 0)
        inp = pt.ones(4)
        block.assert_not_called()
        actual = repeat(inp)
        norm.assert_called_once_with(inp)
        self.assertIs(actual, expected)

    def test_no_repetitions_norm_after(self):
        block = Mock()
        expected = pt.ones(4) * 2
        norm = Mock(return_value=expected)
        norm_cls = Mock(return_value=norm)
        skip = SkipConnection(block, norm_first=False, norm_cls=norm_cls)
        repeat = Repeat(skip, 0)
        inp = pt.ones(4)
        block.assert_not_called()
        actual = repeat(inp)
        norm.assert_not_called()
        self.assertIs(actual, inp)


if __name__ == '__main__':
    unittest.main()

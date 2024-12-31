import unittest
from unittest.mock import Mock
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import SkipConnection
from swak.pt.misc import Identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.skip = SkipConnection(Identity())

    def test_has_block(self):
        self.assertTrue(hasattr(self.skip, 'block'))

    def test_block(self):
        self.assertIsInstance(self.skip.block, Identity)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.skip, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.skip.drop, ptn.Dropout)

    def test_has_norm_first(self):
        self.assertTrue(hasattr(self.skip, 'norm_first'))

    def test_norm_first(self):
        self.assertIsInstance(self.skip.norm_first, bool)
        self.assertTrue(self.skip.norm_first)

    def test_has_norm_cls(self):
        self.assertTrue(hasattr(self.skip, 'norm_cls'))

    def test_norm_cls(self):
        self.assertIs(self.skip.norm_cls, Identity)

    def test_has_args(self):
        self.assertTrue(hasattr(self.skip, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.skip.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.skip, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.skip.kwargs)

    def test_has_norm(self):
        self.assertTrue(hasattr(self.skip, 'norm'))

    def test_norm(self):
        self.assertIsInstance(self.skip.norm, Identity)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.skip, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.skip.reset_parameters))

    def test_reset_parameters_called(self):
        block = Mock()
        block.new = Mock(return_value=block)
        norm = Mock()
        norm_cls = Mock(return_value=norm)
        skip = SkipConnection(block, norm_cls=norm_cls)
        skip.reset_parameters()
        block.reset_parameters.assert_called_once_with()
        norm.reset_parameters.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.skip, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.skip.new))

    def test_call_new(self):
        new = self.skip.new()
        self.assertIsInstance(new, SkipConnection)
        self.assertIsInstance(new.block, Identity)
        self.assertIs(self.skip.drop, new.drop)
        self.assertIs(self.skip.norm_cls, new.norm_cls)
        self.assertDictEqual(self.skip.kwargs, new.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.skip))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.skip = SkipConnection(
            Identity(),
            ptn.AlphaDropout(),
            False,
            ptn.BatchNorm1d,
            4,
            affine=False
        )

    def test_drop(self):
        self.assertIsInstance(self.skip.drop, ptn.AlphaDropout)

    def test_norm_first(self):
        self.assertFalse(self.skip.norm_first)

    def test_norm_cls(self):
        self.assertIs(self.skip.norm_cls, ptn.BatchNorm1d)

    def test_args(self):
        self.assertTupleEqual((4,), self.skip.args)

    def test_kwargs(self):
        self.assertDictEqual({'affine': False}, self.skip.kwargs)

    def test_norm(self):
        self.assertIsInstance(self.skip.norm, ptn.BatchNorm1d)
        self.assertEqual(4, self.skip.norm.num_features)
        self.assertFalse(self.skip.norm.affine)


class TestUsage(unittest.TestCase):

    def test_norm_first(self):
        inp = pt.ones(4)
        norm_out = pt.ones(4) * 2.0
        block_out = pt.ones(4) * 3.0
        drop_out = pt.ones (4) * 4.0
        norm = Mock(return_value=norm_out)
        norm_cls = Mock(return_value=norm)
        block = Mock(return_value=block_out)
        drop = Mock(return_value=drop_out)
        skip = SkipConnection(block, drop, True, norm_cls)
        actual = skip(inp)
        norm.assert_called_once_with(inp)
        block.assert_called_once_with(norm_out)
        drop.assert_called_once_with(block_out)
        pt.testing.assert_close(actual, pt.ones(4) * 2.5)

    def test_norm_after(self):
        inp = pt.ones(4)
        norm_out = pt.ones(4) * 2.0
        block_out = pt.ones(4) * 3.0
        drop_out = pt.ones(4) * 4.0
        norm = Mock(return_value=norm_out)
        norm_cls = Mock(return_value=norm)
        block = Mock(return_value=block_out)
        drop = Mock(return_value=drop_out)
        skip = SkipConnection(block, drop, False, norm_cls)
        actual = skip(inp)
        block.assert_called_once_with(inp)
        drop.assert_called_once_with(block_out)
        pt.testing.assert_close(actual, norm_out)


if __name__ == '__main__':
    unittest.main()

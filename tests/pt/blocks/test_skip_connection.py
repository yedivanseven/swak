import unittest
from unittest.mock import Mock, patch
import torch as pt
import torch.nn as ptn
from swak.pt.blocks import SkipConnection, ActivatedBlock
from swak.pt.misc import Identity


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.skip = SkipConnection(ActivatedBlock(4))

    def test_has_block(self):
        self.assertTrue(hasattr(self.skip, 'block'))

    def test_block(self):
        self.assertIsInstance(self.skip.block, ActivatedBlock)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.skip, 'dropout'))

    def test_dropout(self):
        self.assertIsInstance(self.skip.dropout, float)
        self.assertEqual(0.0, self.skip.dropout)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.skip, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.skip.drop, ptn.Dropout)
        self.assertEqual(0.0, self.skip.drop.p)

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

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.skip, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.skip.mod_dim, int)
        self.assertEqual(4, self.skip.mod_dim)

    def test_has_device(self):
        self.assertTrue(hasattr(self.skip, 'device'))

    def test_device(self):
        self.assertEqual(pt.device('cpu'), self.skip.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.skip, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.skip.dtype, pt.float)
        self.assertIs(self.skip.block.dtype, self.skip.dtype)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.skip, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.skip.reset_parameters))

    def test_reset_parameters_called_on_block(self):
        with patch.object(self.skip.block, 'reset_parameters') as mock:
            self.skip.reset_parameters()
            mock.assert_called_once_with()

    def test_reset_parameters_called_on_norm(self):
        with patch.object(self.skip.norm, 'reset_parameters') as mock:
            self.skip.reset_parameters()
            mock.assert_called_once_with()

    def test_has_new(self):
        self.assertTrue(hasattr(self.skip, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.skip.new))

    def test_call_new(self):
        new = self.skip.new()
        self.assertIsInstance(new, SkipConnection)
        self.assertIsInstance(new.block, ActivatedBlock)
        self.assertIsNot(self.skip.block, new.block)
        self.assertEqual(self.skip.mod_dim, new.mod_dim)
        self.assertIs(self.skip.dropout, new.dropout)
        self.assertEqual(self.skip.device, new.device)
        self.assertIs(self.skip.dtype, new.dtype)
        self.assertIsNot(self.skip.drop, new.drop)
        self.assertIs(self.skip.norm_cls, new.norm_cls)
        self.assertIsNot(self.skip.norm, new.norm)
        self.assertTupleEqual(self.skip.args, new.args)
        self.assertDictEqual(self.skip.kwargs, new.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.skip))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.skip = SkipConnection(
            ActivatedBlock(4),
            0.1,
            False,
            ptn.BatchNorm1d,
            4,
            dtype=pt.float64,
            affine=False,
        )

    def test_dropout(self):
        self.assertEqual(0.1, self.skip.dropout)

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

    def test_dtype(self):
        self.assertIs(self.skip.dtype, pt.float64)
        self.assertIs(self.skip.block.dtype, self.skip.dtype)


class TestUsage(unittest.TestCase):

    @patch('torch.nn.Dropout.forward')
    def test_norm_first(self, drop):
        inp = pt.ones(4)
        norm_out = pt.ones(4) * 2.0
        block_out = pt.ones(4) * 3.0
        drop_out = pt.ones (4) * 4.0
        norm = Mock(return_value=norm_out)
        norm_cls = Mock(return_value=norm)
        block = Mock(return_value=block_out)
        block.to.return_value = block
        drop.return_value = drop_out
        skip = SkipConnection(block, norm_cls=norm_cls, norm_first=True)
        actual = skip(inp)
        norm.assert_called_once_with(inp)
        block.assert_called_once_with(norm_out)
        drop.assert_called_once_with(block_out)
        pt.testing.assert_close(actual, inp + drop_out)

    @patch('torch.nn.Dropout.forward')
    def test_norm_after(self, drop):
        inp = pt.ones(4)
        norm_out = pt.ones(4) * 2.0
        block_out = pt.ones(4) * 3.0
        drop_out = pt.ones(4) * 4.0
        norm = Mock(return_value=norm_out)
        norm_cls = Mock(return_value=norm)
        block = Mock(return_value=block_out)
        block.to.return_value = block
        drop.return_value = drop_out
        skip = SkipConnection(block, norm_cls=norm_cls, norm_first=False)
        actual = skip(inp)
        block.assert_called_once_with(inp)
        drop.assert_called_once_with(block_out)
        pt.testing.assert_close(norm.call_args[0][0], inp + drop_out)
        pt.testing.assert_close(actual, norm_out)


if __name__ == '__main__':
    unittest.main()

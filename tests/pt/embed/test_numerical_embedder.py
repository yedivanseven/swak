import unittest
from unittest.mock import patch
import torch as pt
import torch.nn as ptn
from swak.pt.embed import NumericalEmbedder, ActivatedEmbedder, GatedEmbedder


class EmbCls(ptn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args):
        return args[0] if len(args) == 1 else args

    def reset_parameters(self):
        pass


class NewEmbCls(ptn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args):
        return args[0] if len(args) == 1 else args


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = NumericalEmbedder(4, 2, EmbCls, 42, foo='bar')

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.embed, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.embed.n_features, int)
        self.assertEqual(2, self.embed.n_features)

    def test_has_emb_cls(self):
        self.assertTrue(hasattr(self.embed, 'emb_cls'))

    def test_emb_cls(self):
        self.assertIs(self.embed.emb_cls, EmbCls)

    def test_has_args(self):
        self.assertTrue(hasattr(self.embed, 'args'))

    def test_args(self):
        self.assertTupleEqual((42,), self.embed.args)

    def test_default_args(self):
        embed = NumericalEmbedder(4, 2, EmbCls)
        self.assertTupleEqual((), embed.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.embed, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({'foo': 'bar'}, self.embed.kwargs)

    def test_default_kwargs(self):
        embed = NumericalEmbedder(4, 2, EmbCls)
        self.assertDictEqual({}, embed.kwargs)

    def test_has_embed(self):
        self.assertTrue(hasattr(self.embed, 'embed'))

    def test_embed(self):
        self.assertIsInstance(self.embed.embed, ptn.ModuleList)
        self.assertEqual(2, len(self.embed.embed))
        e1, e2 = self.embed.embed
        self.assertIsInstance(e1, EmbCls)
        self.assertIsInstance(e2, EmbCls)
        self.assertTupleEqual((4, 42), e1.args)
        self.assertTupleEqual((4, 42), e2.args)
        self.assertDictEqual({'foo': 'bar'}, e1.kwargs)
        self.assertDictEqual({'foo': 'bar'}, e2.kwargs)

    def test_has_features(self):
        self.assertTrue(hasattr(self.embed, 'features'))

    def test_features(self):
        self.assertIsInstance(self.embed.features, range)
        self.assertListEqual([0, 1], list(self.embed.features))

    def test_has_dim(self):
        self.assertTrue(hasattr(self.embed, 'dim'))

    def test_dim_non_zero_features(self):
        self.assertIsInstance(self.embed.dim, int)
        self.assertEqual(-2, self.embed.dim)

    def test_dim_zero_features(self):
        embed = NumericalEmbedder(4, 0, EmbCls, foo='bar')
        self.assertIsInstance(embed.dim, int)
        self.assertEqual(-1, embed.dim)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch.object(EmbCls, 'reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        self.assertEqual(2, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new_defaults(self):
        new = self.embed.new()
        self.assertIsInstance(new, NumericalEmbedder)
        self.assertIsNot(new, self.embed)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertEqual(self.embed.n_features, new.n_features)
        self.assertIs(new.emb_cls, self.embed.emb_cls)
        self.assertTupleEqual((), new.args)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)

    def test_call_new_update(self):
        new = self.embed.new(8, 5, NewEmbCls, 'answer', foo=42, bar='baz')
        self.assertIsInstance(new, NumericalEmbedder)
        self.assertEqual(8, new.mod_dim)
        self.assertEqual(5, new.n_features)
        self.assertIs(new.emb_cls, NewEmbCls)
        self.assertTupleEqual(('answer',), new.args)
        self.assertDictEqual({'foo': 42, 'bar': 'baz'}, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.embed = NumericalEmbedder(4, 2, ActivatedEmbedder)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    @patch.object(EmbCls, 'forward')
    def test_emb_cls_called(self, forward):
        forward.return_value = pt.ones(5, 3, 4)
        embed = NumericalEmbedder(4, 2, EmbCls)
        inp = pt.ones(5, 3, 2)
        inp[:, :, 1] = 2
        _ = embed(inp)
        ((arg1,), _), ((arg2,), _) = forward.call_args_list
        pt.testing.assert_close(arg1, pt.ones(5, 3, 1))
        pt.testing.assert_close(arg2, 2 * pt.ones(5, 3, 1))

    def test_1d(self):
        inp = pt.ones(2)
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([2, 4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 2)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 2, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(1, 3, 2)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([1, 3, 2, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(5, 1, 3, 2)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 1, 3, 2, 4]), actual.shape)

    def test_empty_features(self):
        inp = pt.ones(5, 0, 2)
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 0, 2, 4]), actual.shape)

    def test_no_features(self):
        embed = NumericalEmbedder(4, 0, GatedEmbedder)
        inp = pt.ones(5, 3, 0)
        actual = embed(inp)
        self.assertEqual(pt.Size([5, 3, 0, 4]), actual.shape)


if __name__ == '__main__':
    unittest.main()

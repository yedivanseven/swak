import unittest
from unittest.mock import patch
import torch as pt
import torch.nn as ptn
from swak.pt.embed import CategoricalEmbedder


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed = CategoricalEmbedder(
            2,
            (5, 6),
            7, 8,
            scale_grad_by_freq=True
        )

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(2, self.embed.mod_dim)

    def test_has_cat_counts(self):
        self.assertTrue(hasattr(self.embed, 'cat_counts'))

    def test_cat_counts(self):
        self.assertTupleEqual((5, 6, 7, 8), self.embed.cat_counts)

    def test_cat_counts_default(self):
        embed = CategoricalEmbedder(2)
        self.assertTupleEqual((), embed.cat_counts)

    def test_cat_counts_zero_features(self):
        embed = CategoricalEmbedder(2, [])
        self.assertTupleEqual((), embed.cat_counts)

    def test_cat_count_integer(self):
        embed = CategoricalEmbedder(2, 3)
        self.assertTupleEqual((3,), embed.cat_counts)

    def test_cat_count_integer_cat_counts(self):
        embed = CategoricalEmbedder(2, 3, 4)
        self.assertTupleEqual((3, 4), embed.cat_counts)

    def test_has_device(self):
        self.assertTrue(hasattr(self.embed, 'device'))

    def test_device(self):
        self.assertEqual(self.embed.device, pt.device('cpu'))

    def test_device_zero_features(self):
        embed = CategoricalEmbedder(4, device='cpu')
        self.assertIsNone(embed.device)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.embed, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.embed.dtype, pt.float)

    def test_dtype_zero_features(self):
        embed = CategoricalEmbedder(4, dtype=pt.float64)
        self.assertIsNone(embed.dtype)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.embed, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({'scale_grad_by_freq': True}, self.embed.kwargs)

    def test_has_embed(self):
        self.assertTrue(hasattr(self.embed, 'embed'))

    @patch('torch.nn.Embedding', return_value=pt.nn.Linear(1, 4))
    def test_embedding_called(self, emb):
        embed = CategoricalEmbedder(4, [3], sparse=True)
        self.assertTrue(hasattr(embed, 'kwargs'))
        self.assertDictEqual({'sparse': True}, embed.kwargs)
        emb.assert_called_once_with(3, 4, sparse=True)

    @patch('torch.nn.ModuleList.to')
    def test_to_called(self, to):
        _ = CategoricalEmbedder(4, [3], dtype='bar')
        to.assert_called_once_with(device='cpu', dtype='bar')

    def test_embed(self):
        self.assertIsInstance(self.embed.embed, ptn.ModuleList)
        self.assertEqual(4, len(self.embed.embed))
        e1, e2, e3, e4 = self.embed.embed
        self.assertIsInstance(e1, ptn.Embedding)
        self.assertIsInstance(e2, ptn.Embedding)
        self.assertIsInstance(e3, ptn.Embedding)
        self.assertIsInstance(e4, ptn.Embedding)
        self.assertEqual(5, e1.num_embeddings)
        self.assertEqual(6, e2.num_embeddings)
        self.assertEqual(7, e3.num_embeddings)
        self.assertEqual(8, e4.num_embeddings)
        self.assertEqual(2, e1.embedding_dim)
        self.assertEqual(2, e2.embedding_dim)
        self.assertEqual(2, e3.embedding_dim)
        self.assertEqual(2, e4.embedding_dim)
        self.assertTrue(e1.scale_grad_by_freq)
        self.assertTrue(e2.scale_grad_by_freq)
        self.assertTrue(e3.scale_grad_by_freq)
        self.assertTrue(e4.scale_grad_by_freq)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.embed, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.embed.n_features, int)
        self.assertEqual(4, self.embed.n_features)

    def test_has_features(self):
        self.assertTrue(hasattr(self.embed, 'features'))

    def test_features(self):
        self.assertIsInstance(self.embed.features, range)
        self.assertListEqual([0, 1, 2,3 ], list(self.embed.features))

    def test_has_dim(self):
        self.assertTrue(hasattr(self.embed, 'dim'))

    def test_dim_non_zero_features(self):
        self.assertIsInstance(self.embed.dim, int)
        self.assertEqual(-2, self.embed.dim)

    def test_dim_zero_features(self):
        embed = CategoricalEmbedder(2)
        self.assertIsInstance(embed.dim, int)
        self.assertEqual(-1, embed.dim)

    def test_bool_non_zero_features(self):
        self.assertTrue(self.embed)

    def test_bool_zero_features(self):
        embed = CategoricalEmbedder(2)
        self.assertFalse(embed)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    @patch('torch.nn.Embedding.reset_parameters')
    def test_reset_parameters_called(self, mock):
        self.embed.reset_parameters()
        self.assertEqual(4, mock.call_count)

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new(self):
        new = self.embed.new()
        self.assertIsInstance(new, CategoricalEmbedder)
        self.assertIsNot(new, self.embed)
        self.assertEqual(self.embed.device, new.device)
        self.assertEqual(self.embed.dtype, new.dtype)
        self.assertEqual(self.embed.mod_dim, new.mod_dim)
        self.assertTupleEqual(self.embed.cat_counts, new.cat_counts)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)

    def test_call_new_no_features(self):
        embed = CategoricalEmbedder(2)
        new = embed.new()
        self.assertIsInstance(new, CategoricalEmbedder)
        self.assertIsNot(new, embed)
        self.assertEqual(embed.device, new.device)
        self.assertEqual(embed.dtype, new.dtype)
        self.assertEqual(embed.mod_dim, new.mod_dim)
        self.assertTupleEqual(embed.cat_counts, new.cat_counts)
        self.assertDictEqual(embed.kwargs, new.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.embed = CategoricalEmbedder(4, 3, 2)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(2, device='cpu').long()
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([2, 4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 2, device='cpu').long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 2, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(1, 3, 2, device='cpu').long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([1, 3, 2, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(5, 1, 3, 2, device='cpu').long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 1, 3, 2, 4]), actual.shape)

    def test_empty_features(self):
        inp = pt.ones(5, 0, 2, device='cpu').long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 0, 2, 4]), actual.shape)

    def test_no_features(self):
        embed = CategoricalEmbedder(4)
        inp = pt.ones(5, 3, 0, device='cpu').long()
        actual = embed(inp)
        self.assertEqual(pt.Size([5, 3, 0, 4]), actual.shape)


if __name__ == '__main__':
    unittest.main()

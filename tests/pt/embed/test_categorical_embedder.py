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

    def test_has_out_dim(self):
        self.assertTrue(hasattr(self.embed, 'out_dim'))

    def test_out_dim(self):
        self.assertIsInstance(self.embed.out_dim, int)
        self.assertEqual(2, self.embed.out_dim)

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

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new_defaults(self):
        new = self.embed.new()
        self.assertIsInstance(new, CategoricalEmbedder)
        self.assertIsNot(new, self.embed)
        self.assertEqual(self.embed.out_dim, new.out_dim)
        self.assertTupleEqual(self.embed.cat_counts, new.cat_counts)
        self.assertDictEqual(self.embed.kwargs, new.kwargs)

    def test_call_new_updates(self):
        new = self.embed.new(
            8,
            (1, 2),
            3,
            4,
            scale_grad_by_freq=False,
            max_norm=1
        )
        self.assertIsInstance(new, CategoricalEmbedder)
        self.assertEqual(8, new.out_dim)
        self.assertTupleEqual((1, 2, 3, 4), new.cat_counts)
        self.assertDictEqual(
            {'scale_grad_by_freq': False, 'max_norm': 1},
            new.kwargs
        )


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.embed = CategoricalEmbedder(4, 3, 2)

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(2).long()
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([2, 4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 2).long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 2, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(1, 3, 2).long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([1, 3, 2, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(5, 1, 3, 2).long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 1, 3, 2, 4]), actual.shape)

    def test_empty_features(self):
        inp = pt.ones(5, 0, 2).long()
        actual = self.embed(inp)
        self.assertEqual(pt.Size([5, 0, 2, 4]), actual.shape)

    def test_no_features(self):
        embed = CategoricalEmbedder(4)
        inp = pt.ones(5, 3, 0).long()
        actual = embed(inp)
        self.assertEqual(pt.Size([5, 3, 0, 4]), actual.shape)


if __name__ == '__main__':
    unittest.main()

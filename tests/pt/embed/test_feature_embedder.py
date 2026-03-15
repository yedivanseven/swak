import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.exceptions import EmbeddingError
from swak.pt.embed import (
    ActivatedEmbedder,
    GatedEmbedder,
    NumericalEmbedder,
    CategoricalEmbedder,
    FeatureEmbedder
)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.embed_num = NumericalEmbedder(4, 2, ActivatedEmbedder)
        self.embed_cat = CategoricalEmbedder(4, 6, 7, 8)
        self.embed = FeatureEmbedder(self.embed_num, self.embed_cat)

    def test_has_embed_num(self):
        self.assertTrue(hasattr(self.embed, 'embed_num'))

    def test_embed_num(self):
        self.assertIs(self.embed.embed_num, self.embed_num)

    def test_has_embed_cat(self):
        self.assertTrue(hasattr(self.embed, 'embed_cat'))

    def test_embed_cat(self):
        self.assertIs(self.embed.embed_cat, self.embed_cat)

    def test_to_called(self):
        with patch.object(
                self.embed_num, 'to', return_value=self.embed_num
        ) as num_to, patch.object(
                self.embed_cat, 'to', return_value=self.embed_cat
        ) as cat_to:
            _ = FeatureEmbedder(self.embed_num, self.embed_cat, dtype='bar')
            num_to.assert_called_once_with(device='cpu', dtype='bar')
            cat_to.assert_called_once_with(device='cpu', dtype='bar')

    def test_has_device(self):
        self.assertTrue(hasattr(self.embed, 'device'))

    def test_device(self):
        self.assertEqual(self.embed.device, pt.device('cpu'))

    def test_device_no_num_features(self):
        num = NumericalEmbedder(4, 0, ActivatedEmbedder)
        cat = CategoricalEmbedder(4, 6, 7, 8)
        embed = FeatureEmbedder(num, cat)
        self.assertEqual(pt.device('cpu'), embed.device)

    def test_device_no_cat_features(self):
        num = NumericalEmbedder(4, 2, ActivatedEmbedder)
        cat = CategoricalEmbedder(4)
        embed = FeatureEmbedder(num, cat)
        self.assertEqual(pt.device('cpu'), embed.device)

    def test_device_no_features(self):
        num = NumericalEmbedder(4, 0, ActivatedEmbedder)
        cat = CategoricalEmbedder(4)
        embed = FeatureEmbedder(num, cat)
        self.assertIsNone(embed.dtype)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.embed, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.embed.dtype, pt.float)

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.embed, 'mod_dim'))

    def test_dtype_no_num_features(self):
        num = NumericalEmbedder(4, 0, ActivatedEmbedder, dtype=pt.float64)
        cat = CategoricalEmbedder(4, 6, 7, 8, dtype=pt.float16)
        embed = FeatureEmbedder(num, cat)
        self.assertIs(embed.dtype, pt.float)

    def test_dtype_no_cat_features(self):
        num = NumericalEmbedder(4, 2, ActivatedEmbedder, dtype=pt.float64)
        cat = CategoricalEmbedder(4, dtype=pt.float64)
        embed = FeatureEmbedder(num, cat)
        self.assertIs(embed.dtype, pt.float)

    def test_dtype_no_features(self):
        num = NumericalEmbedder(4, 0, ActivatedEmbedder, dtype=pt.float64)
        cat = CategoricalEmbedder(4, dtype=pt.float16)
        embed = FeatureEmbedder(num, cat)
        self.assertIsNone(embed.dtype)

    def test_mod_dim(self):
        self.assertIsInstance(self.embed.mod_dim, int)
        self.assertEqual(4, self.embed.mod_dim)

    def test_has_n_num(self):
        self.assertTrue(hasattr(self.embed, 'n_num'))

    def test_n_num(self):
        self.assertIsInstance(self.embed.n_num, int)
        self.assertEqual(2, self.embed.n_num)

    def test_has_n_cat(self):
        self.assertTrue(hasattr(self.embed, 'n_cat'))

    def test_n_cat(self):
        self.assertIsInstance(self.embed.n_cat, int)
        self.assertEqual(3, self.embed.n_cat)

    def test_has_n_features(self):
        self.assertTrue(hasattr(self.embed, 'n_features'))

    def test_n_features(self):
        self.assertIsInstance(self.embed.n_features, int)
        self.assertEqual(5, self.embed.n_features)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.embed, 'reset_parameters'))

    def test_reset_parameters(self):
        self.assertTrue(callable(self.embed.reset_parameters))

    def test_call_reset_parameters(self):
        self.embed.reset_parameters()

    def test_has_new(self):
        self.assertTrue(hasattr(self.embed, 'new'))

    def test_new(self):
        self.assertTrue(callable(self.embed.new))

    def test_call_new_defaults(self):
        new = self.embed.new()
        self.assertIsInstance(new, FeatureEmbedder)
        self.assertIsNot(new, self.embed)
        self.assertIsInstance(new.embed_num, NumericalEmbedder)
        self.assertIsNot(new.embed_num, self.embed_num)
        self.assertIsInstance(new.embed_cat, CategoricalEmbedder)
        self.assertIsNot(new.embed_cat, self.embed_cat)
        self.assertEqual(4, new.embed_num.mod_dim)
        self.assertEqual(2, new.embed_num.n_features)
        self.assertIs(new.embed_num.emb_cls, ActivatedEmbedder)
        self.assertEqual(4, new.embed_cat.mod_dim)
        self.assertTupleEqual((6, 7, 8), new.embed_cat.cat_counts)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.embed_num = NumericalEmbedder(
            4,
            2,
            ActivatedEmbedder
        )
        self.embed_cat = CategoricalEmbedder(
            4,
            6,
            7,
            8
        )
        self.embed = FeatureEmbedder(
            self.embed_num,
            self.embed_cat
        )

    def test_callable(self):
        self.assertTrue(callable(self.embed))

    def test_1d(self):
        inp = pt.ones(5, device='cpu')
        actual = self.embed(inp)
        self.assertIsInstance(actual, pt.Tensor)
        self.assertEqual(pt.Size([5, 4]), actual.shape)

    def test_2d(self):
        inp = pt.ones(3, 5, device='cpu')
        actual = self.embed(inp)
        self.assertEqual(pt.Size([3, 5, 4]), actual.shape)

    def test_3d(self):
        inp = pt.ones(2, 3, 5, device='cpu')
        actual = self.embed(inp)
        self.assertEqual(pt.Size([2, 3, 5, 4]), actual.shape)

    def test_4d(self):
        inp = pt.ones(1, 2, 3, 5, device='cpu')
        actual = self.embed(inp)
        self.assertEqual(pt.Size([1, 2, 3, 5, 4]), actual.shape)

    def test_no_num_features(self):
        embed_num = NumericalEmbedder(4, 0, GatedEmbedder)
        embed = FeatureEmbedder(embed_num, self.embed_cat)
        inp = pt.ones(1, 2, 3, device='cpu')
        actual = embed(inp)
        self.assertEqual(pt.Size([1, 2, 3, 4]), actual.shape)

    def test_no_cat_features(self):
        embed_cat = CategoricalEmbedder(4, [])
        embed = FeatureEmbedder(self.embed_num, embed_cat)
        inp = pt.ones(5, 3, 2, device='cpu')
        actual = embed(inp)
        self.assertEqual(pt.Size([5, 3, 2, 4]), actual.shape)

    def test_empty_features(self):
        inp = pt.ones(2, 0, 5, device='cpu')
        actual = self.embed(inp)
        self.assertEqual(pt.Size([2, 0, 5, 4]), actual.shape)

    def test_no_features(self):
        embed_num = NumericalEmbedder(4, 0, ActivatedEmbedder)
        embed_cat = CategoricalEmbedder(4, [])
        embed = FeatureEmbedder(embed_num, embed_cat)
        inp = pt.ones(5, 3, 0)
        actual = embed(inp)
        self.assertEqual(pt.Size([5, 3, 0, 4]), actual.shape)

    @patch.object(CategoricalEmbedder, 'forward')
    @patch.object(NumericalEmbedder, 'forward')
    def test_forwards_called(self, num, cat):
        inp = pt.ones(2, 3, 5)
        num.return_value = pt.ones(2, 3, 2, 4)
        cat.return_value = pt.ones(2, 3, 3, 4)
        _ = self.embed(inp)
        expected = pt.ones(2, 3, 2)
        (actual,), _ = num.call_args
        pt.testing.assert_close(actual, expected)
        expected = pt.ones(2, 3, 3).long()
        (actual,), _ = cat.call_args
        pt.testing.assert_close(actual, expected)

    def test_raises_on_embedding_dim_mismatch(self):
        embed_num = NumericalEmbedder(5, 2, GatedEmbedder)
        with self.assertRaises(EmbeddingError):
            _ = FeatureEmbedder(embed_num, self.embed_cat)


if __name__ == '__main__':
    unittest.main()

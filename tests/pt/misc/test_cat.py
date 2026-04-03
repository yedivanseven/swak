import unittest
from unittest.mock import patch
from swak.pt.misc import Cat


class TestCat(unittest.TestCase):

    def test_has_dim(self):
        cat = Cat()
        self.assertTrue(hasattr(cat, 'dim'))

    def test_default_dim(self):
        cat = Cat()
        self.assertIsInstance(cat.dim, int)
        self.assertEqual(0, cat.dim)

    def test_custom_dim(self):
        cat = Cat(2)
        self.assertEqual(2, cat.dim)

    def test_callable(self):
        self.assertTrue(callable(Cat()))

    @patch('torch.cat')
    def test_cat_called_with_default_dim(self,  mock):
        cat = Cat()
        _ = cat([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=0)

    @patch('torch.cat')
    def test_cat_called_with_custom_dim(self,  mock):
        cat = Cat(2)
        _ = cat([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=2)

    def test_default_repr(self):
        expected = 'Cat(0)'
        self.assertEqual(expected, repr(Cat()))

    def test_custom_repr(self):
        expected = 'Cat(2)'
        self.assertEqual(expected, repr(Cat(2)))


if __name__ == '__main__':
    unittest.main()

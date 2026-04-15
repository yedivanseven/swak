import unittest
from unittest.mock import patch
from swak.pt.misc import Stack


class TestStack(unittest.TestCase):

    def test_has_dim(self):
        stack = Stack()
        self.assertTrue(hasattr(stack, 'dim'))

    def test_default_dim(self):
        stack = Stack()
        self.assertIsInstance(stack.dim, int)
        self.assertEqual(0, stack.dim)

    def test_custom_dim(self):
        stack = Stack(2)
        self.assertEqual(2, stack.dim)

    def test_callable(self):
        self.assertTrue(callable(Stack()))

    @patch('torch.stack')
    def test_cat_called_with_default_dim(self,  mock):
        stack = Stack()
        _ = stack([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=0)

    @patch('torch.stack')
    def test_cat_called_with_custom_dim(self,  mock):
        stack = Stack(2)
        _ = stack([1, 2, 3])
        mock.assert_called_once_with([1, 2, 3], dim=2)

    def test_default_repr(self):
        expected = 'Stack(0)'
        self.assertEqual(expected, repr(Stack()))

    def test_custom_repr(self):
        expected = 'Stack(2)'
        self.assertEqual(expected, repr(Stack(2)))


if __name__ == '__main__':
    unittest.main()

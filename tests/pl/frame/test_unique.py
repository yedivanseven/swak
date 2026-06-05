import pickle
import unittest
from unittest.mock import Mock
from swak.pl import Unique


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.unique = Unique()

    def test_has_subset(self):
        self.assertTrue(hasattr(self.unique, 'subset'))

    def test_subset(self):
        self.assertIsNone(self.unique.subset)

    def test_has_keep(self):
        self.assertTrue(hasattr(self.unique, 'keep'))

    def test_keep(self):
        self.assertEqual('any', self.unique.keep)

    def test_has_maintain_order(self):
        self.assertTrue(hasattr(self.unique, 'maintain_order'))

    def test_maintain_order(self):
        self.assertIsInstance(self.unique.maintain_order, bool)
        self.assertFalse(self.unique.maintain_order)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.subset = ['foo', 'bar']
        self.keep = 'last'
        self.maintain_order = True
        self.unique = Unique(
            self.subset,
            keep=self.keep,
            maintain_order=self.maintain_order
        )

    def test_subset(self):
        self.assertListEqual(self.subset, self.unique.subset)

    def test_keep(self):
        self.assertEqual('last', self.unique.keep)

    def test_maintain_order(self):
        self.assertIsInstance(self.unique.maintain_order, bool)
        self.assertTrue(self.unique.maintain_order)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.subset = ['foo', 'bar']
        self.keep = 'last'
        self.maintain_order = True
        self.unique = Unique(
            self.subset,
            keep=self.keep,
            maintain_order=self.maintain_order
        )

    def test_callable(self):
        self.assertTrue(callable(self.unique))

    def test_sort_called(self):
        df = Mock()
        _ = self.unique(df)
        df.unique.assert_called_once_with(
            self.subset,
            keep=self.keep,
            maintain_order=self.maintain_order
        )

    def test_return_value(self):
        df = Mock()
        df.unique = Mock(return_value='answer')
        actual = self.unique(df)
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        unique = Unique()
        expected = "Unique(None, keep='any', maintain_order=False)"
        self.assertEqual(expected, repr(unique))

    def test_custom_repr(self):
        unique = Unique(['foo', 'bar'], keep='last', maintain_order=True)
        expected = "Unique(['foo', 'bar'], keep='last', maintain_order=True)"
        self.assertEqual(expected, repr(unique))

    def test_pickle_works(self):
        unique = Unique(['foo', 'bar'], keep='last', maintain_order=True)
        _ = pickle.loads(pickle.dumps(unique))


if __name__ == '__main__':
    unittest.main()

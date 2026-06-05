import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import DropDuplicates


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = DropDuplicates()

    def test_has_subset(self):
        self.assertTrue(hasattr(self.drop, 'subset'))

    def test_subset(self):
        self.assertIsNone(self.drop.subset)

    def test_has_keep(self):
        self.assertTrue(hasattr(self.drop, 'keep'))

    def test_keep(self):
        self.assertEqual('first', self.drop.keep)

    def test_has_ignore_index(self):
        self.assertTrue(hasattr(self.drop, 'ignore_index'))

    def test_ignore_index(self):
        self.assertIsInstance(self.drop.ignore_index, bool)
        self.assertFalse(self.drop.ignore_index)

class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.subset = ['foo', 'bar']
        self.keep = 'last'
        self.ignore_index = True
        self.drop = DropDuplicates(
            self.subset,
            keep=self.keep,
            ignore_index=self.ignore_index
        )

    def test_subset(self):
        self.assertListEqual(self.subset, self.drop.subset)

    def test_keep(self):
        self.assertEqual('last', self.drop.keep)

    def test_ignore_index(self):
        self.assertIsInstance(self.drop.ignore_index, bool)
        self.assertTrue(self.drop.ignore_index)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.subset = ['foo', 'bar']
        self.keep = 'last'
        self.ignore_index = True
        self.drop = DropDuplicates(
            self.subset,
            keep=self.keep,
            ignore_index=self.ignore_index
        )

    def test_callable(self):
        self.assertTrue(callable(self.drop))

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.drop_duplicates = Mock(return_value='answer')
        actual = self.drop(df)
        df.drop_duplicates.assert_called_once_with(
            self.subset,
            keep='last',
            ignore_index=self.ignore_index
        )
        self.assertEqual('answer', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.drop_duplicates = Mock(return_value='answer')
        actual = self.drop(df)
        df.drop_duplicates.assert_called_once_with(
            keep='last',
            ignore_index=self.ignore_index
        )
        self.assertEqual('answer', actual)

    def test_raises_on_wrong_df(self):
        with self.assertRaises(TypeError):
            self.drop(object())


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = DropDuplicates()
        expected = "DropDuplicates(None, keep='first', ignore_index=False)"
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = DropDuplicates( ['foo', 'bar'], keep='last', ignore_index=True)
        expected = ("DropDuplicates(['foo', 'bar'], keep='last',"
                    " ignore_index=True)")
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = DropDuplicates( ['foo', 'bar'], keep='last', ignore_index=True)
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()

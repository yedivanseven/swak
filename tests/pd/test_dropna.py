import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import DropNA


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = DropNA()

    def test_has_axis(self):
        self.assertTrue(hasattr(self.drop, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.drop.axis, int)
        self.assertEqual(0, self.drop.axis)

    def test_has_how(self):
        self.assertTrue(hasattr(self.drop, 'how'))

    def test_how(self):
        self.assertIsNone(self.drop.how)

    def test_has_thresh(self):
        self.assertTrue(hasattr(self.drop, 'thresh'))

    def test_thresh(self):
        self.assertIsNone(self.drop.thresh)

    def test_has_subset(self):
        self.assertTrue(hasattr(self.drop, 'subset'))

    def test_subset(self):
        self.assertIsNone(self.drop.subset)

    def test_has_ignore_index(self):
        self.assertTrue(hasattr(self.drop, 'ignore_index'))

    def test_ignore_index(self):
        self.assertIsInstance(self.drop.ignore_index, bool)
        self.assertTrue(self.drop.ignore_index)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = DropNA(
            axis=1,
            how='all',
            thresh=2,
            subset=['foo', 'bar'],
            ignore_index=False
        )

    def test_axis(self):
        self.assertEqual(1, self.drop.axis)

    def test_how(self):
        self.assertEqual('all', self.drop.how)

    def test_thresh(self):
        self.assertEqual(2, self.drop.thresh)

    def test_subset(self):
        self.assertListEqual(['foo', 'bar'], self.drop.subset)

    def test_ignore_index(self):
        self.assertFalse(self.drop.ignore_index)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(DropNA()))

    def test_dataframe_default(self):
        drop = DropNA()
        df = pd.DataFrame(range(10))
        df.dropna = Mock(return_value='answer')
        actual = drop(df)
        df.dropna.assert_called_once_with(
            axis=0,
            subset=None,
            inplace=False,
            ignore_index=True
        )
        self.assertEqual('answer', actual)

    def test_dataframe_custom(self):
        drop = DropNA(1, subset=['foo', 'bar'], ignore_index=False)
        df = pd.DataFrame(range(10))
        df.dropna = Mock(return_value='answer')
        actual = drop(df)
        df.dropna.assert_called_once_with(
            axis=1,
            subset=['foo', 'bar'],
            inplace=False,
            ignore_index=False
        )
        self.assertEqual('answer', actual)

    def test_dataframe_how(self):
        drop = DropNA(
            axis=1,
            how='all',
            subset=['foo', 'bar'],
            ignore_index=False,
        )
        df = pd.DataFrame(range(10))
        df.dropna = Mock()
        _ = drop(df)
        df.dropna.assert_called_once_with(
            axis=1,
            how='all',
            subset=['foo', 'bar'],
            inplace=False,
            ignore_index=False
        )

    def test_dataframe_thresh(self):
        drop = DropNA(
            axis=1,
            thresh=2,
            subset=['foo', 'bar'],
            ignore_index=True,
        )
        df = pd.DataFrame(range(10))
        df.dropna = Mock()
        _ = drop(df)
        df.dropna.assert_called_once_with(
            axis=1,
            thresh=2,
            subset=['foo', 'bar'],
            inplace=False,
            ignore_index=True
        )

    def test_dataframe_how_thresh(self):
        drop = DropNA(
            axis=1,
            how='any',
            thresh=2,
            subset=['foo', 'bar'],
            ignore_index=True,
        )
        df = pd.DataFrame(range(10))
        df.dropna = Mock()
        _ = drop(df)
        df.dropna.assert_called_once_with(
            axis=1,
            how='any',
            subset=['foo', 'bar'],
            inplace=False,
            ignore_index=True
        )

    def test_series_default(self):
        drop = DropNA()
        df = pd.Series(range(10))
        df.dropna = Mock(return_value='answer')
        actual = drop(df)
        df.dropna.assert_called_once_with(
            axis=0,
            inplace=False,
            ignore_index=True
        )
        self.assertEqual('answer', actual)

    def test_series_custom(self):
        drop = DropNA(1, subset=['foo', 'bar'], ignore_index=False)
        df = pd.Series(range(10))
        df.dropna = Mock(return_value='answer')
        actual = drop(df)
        df.dropna.assert_called_once_with(
            axis=0,
            inplace=False,
            ignore_index=False
        )
        self.assertEqual('answer', actual)

    def test_raises_on_wrong_type(self):
        drop = DropNA()
        with self.assertRaises(TypeError):
            _ = drop(2)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = DropNA()
        expected = ('DropNA(axis=0, how=None, thresh=None, '
                    'subset=None, ignore_index=True)')
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = DropNA(
            axis=1,
            thresh=2,
            subset=['foo', 'bar'],
            ignore_index=False,
        )
        expected = ("DropNA(axis=1, how=None, thresh=2, "
                    "subset=['foo', 'bar'], ignore_index=False)")
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = DropNA(
            axis=1,
            how='all',
            subset=['foo', 'bar'],
            ignore_index=True,
        )
        _ = pickle.loads(pickle.dumps(drop))


if __name__ == '__main__':
    unittest.main()

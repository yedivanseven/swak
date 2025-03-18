import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Drop


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = Drop()

    def test_has_labels(self):
        self.assertTrue(hasattr(self.drop, 'labels'))

    def test_labels(self):
        self.assertIsNone(self.drop.labels)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.drop, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.drop.axis, int)
        self.assertEqual(1, self.drop.axis)

    def test_has_index(self):
        self.assertTrue(hasattr(self.drop, 'index'))

    def test_index(self):
        self.assertIsNone(self.drop.index)

    def test_has_columns(self):
        self.assertTrue(hasattr(self.drop, 'columns'))

    def test_columns(self):
        self.assertIsNone(self.drop.columns)

    def test_has_level(self):
        self.assertTrue(hasattr(self.drop, 'level'))

    def test_level(self):
        self.assertIsNone(self.drop.level)

    def test_has_errors(self):
        self.assertTrue(hasattr(self.drop, 'errors'))

    def test_errors(self):
        self.assertEqual('raise', self.drop.errors)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.drop = Drop(
           ['foo'],
            'bar',
            axis=0,
            index=['baz'],
            columns=1,
            level=2,
            errors='ignore'
        )

    def test_labels(self):
        self.assertListEqual(['foo', 'bar'], self.drop.labels)

    def test_axis(self):
        self.assertEqual(0, self.drop.axis)

    def test_index(self):
        self.assertListEqual(['baz'], self.drop.index)

    def test_columns(self):
        self.assertEqual(1, self.drop.columns)

    def test_level(self):
        self.assertEqual(2, self.drop.level)

    def test_errors(self):
        self.assertEqual('ignore', self.drop.errors)

    def test_none_and_args(self):
        drop = Drop(None, 'foo', 'bar')
        self.assertListEqual(['foo', 'bar'], drop.labels)

    def test_list_only(self):
        drop = Drop(['foo', 'bar'])
        self.assertListEqual(['foo', 'bar'], drop.labels)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.drop = Drop(
            ['foo'],
            'bar',
            axis=0,
            index=['baz'],
            columns=1,
            level=2,
            errors='ignore'
        )

    def test_callable(self):
        self.assertTrue(callable(self.drop))

    def test_drop_called(self):
        df = Mock()
        _ = self.drop(df)
        df.drop.assert_called_once_with(
            ['foo', 'bar'],
            axis=0,
            index=['baz'],
            columns=1,
            level=2,
            inplace=False,
            errors='ignore'
        )

    def test_return_value(self):
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        drop = Drop(1)
        actual = drop(df)
        expected = df.drop(1, axis=1)
        pd.testing.assert_frame_equal(actual, expected)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        drop = Drop()
        expected = (
            'Drop(None, axis=1, index=None, columns=None, '
            "level=None, errors='raise')"
        )
        self.assertEqual(expected, repr(drop))

    def test_custom_repr(self):
        drop = Drop(
            ['foo'],
            'bar',
            axis=0,
            index=['baz'],
            columns=1,
            level=2,
            errors='ignore',
        )
        expected = ("Drop(['foo', 'bar'], axis=0, index=['baz'], "
                    "columns=1, level=2, errors='ignore')")
        self.assertEqual(expected, repr(drop))

    def test_pickle_works(self):
        drop = Drop()
        _ = pickle.loads(pickle.dumps(drop))

if __name__ == '__main__':
    unittest.main()

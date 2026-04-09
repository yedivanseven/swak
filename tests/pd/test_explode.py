import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Explode


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.explode = Explode()

    def test_has_cols(self):
        self.assertTrue(hasattr(self.explode, 'cols'))

    def test_cols(self):
        self.assertListEqual([], self.explode.cols)

    def test_has_ignore_index(self):
        self.assertTrue(hasattr(self.explode, 'ignore_index'))

    def test_ignore_index(self):
        self.assertIsInstance(self.explode.ignore_index, bool)
        self.assertFalse(self.explode.ignore_index)


class TestAttributes(unittest.TestCase):

    def test_single_col(self):
        explode = Explode('foo')
        self.assertListEqual(['foo'], explode.cols)

    def test_single_col_and_cols(self):
        explode = Explode('foo', 'bar', 'baz')
        self.assertListEqual(['foo', 'bar', 'baz'], explode.cols)

    def test_list_col(self):
        explode = Explode(['foo', 'bar'])
        self.assertListEqual(['foo', 'bar'], explode.cols)

    def test_list_col_and_cols(self):
        explode = Explode(['foo', 'bar'], 'baz')
        self.assertListEqual(['foo', 'bar', 'baz'], explode.cols)

    def test_ignore_index(self):
        explode = Explode(['foo', 'bar'], 'baz', ignore_index=True)
        self.assertIsInstance(explode.ignore_index, bool)
        self.assertTrue(explode.ignore_index)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.explode = Explode(['foo', 'bar'], 'baz', ignore_index=True)

    def test_explode_called(self):
        df = pd.DataFrame(range(10))
        df.explode = Mock(return_value='expected')
        actual = self.explode(df)
        df.explode.assert_called_once_with(
            ['foo', 'bar', 'baz'],
            ignore_index=True
        )
        self.assertEqual('expected', actual)

    def test_dataframe(self):
        explode = Explode(0)
        df = pd.DataFrame(range(10))
        _ = explode(df)

    def test_series(self):
        explode = Explode()
        df = pd.Series(range(10))
        _ = explode(df)

    def test_raises_on_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.explode(2)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        explode = Explode()
        expected = 'Explode(ignore_index=False)'
        self.assertEqual(expected, repr(explode))

    def test_custom_repr(self):
        explode = Explode(['foo', 'bar'], 'baz', ignore_index=True)
        expected = "Explode('foo', 'bar', 'baz', ignore_index=True)"
        self.assertEqual(expected, repr(explode))

    def test_pickle_works(self):
        explode = Explode(['foo', 'bar'], 'baz', ignore_index=True)
        _ = pickle.loads(pickle.dumps(explode))


if __name__ == '__main__':
    unittest.main()

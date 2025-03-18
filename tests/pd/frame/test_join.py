import pickle
import pandas as pd
import unittest
from unittest.mock import Mock
from swak.pd import Join


class TestAttributes(unittest.TestCase):

    def test_empty_has_args(self):
        join = Join()
        self.assertTrue(hasattr(join, 'args'))

    def test_empty_has_kwargs(self):
        join = Join()
        self.assertTrue(hasattr(join, 'kwargs'))

    def test_empty_args(self):
        join = Join()
        self.assertTupleEqual((), join.args)

    def test_empty_kwargs(self):
        join = Join()
        self.assertDictEqual({}, join.kwargs)

    def test_args(self):
        join = Join(1, 2, three=3, four=4)
        self.assertTupleEqual((1, 2), join.args)

    def test_kwargs(self):
        join = Join(1, 2, three=3, four=4)
        self.assertDictEqual({'three': 3, 'four': 4}, join.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mock = Mock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])
        self.other_df = pd.DataFrame([[11], [12], [13], [14]], columns=[2])
        self.other_series = pd.Series([21, 22, 23, 24], name=3)

    def test_callable(self):
        join = Join(1, 2, three=3, four=4)
        self.assertTrue(callable(join))

    def test_join_called(self):
        obj = object()
        join = Join(1, 2, three=3, four=4)
        _ = join(self.mock, obj)
        self.mock.join.assert_called_once_with(obj, 1, 2, three=3, four=4)

    def test_join_frame_index(self):
        join = Join()
        actual = join(self.df, self.other_df)
        expected = self.df.join(self.other_df)
        pd.testing.assert_frame_equal(expected, actual)

    def test_join_frame_column(self):
        join = Join(0)
        actual = join(self.df, self.other_df)
        expected = self.df.join(self.other_df, 0)
        pd.testing.assert_frame_equal(expected, actual)

    def test_join_series_index(self):
        join = Join()
        actual = join(self.df, self.other_series)
        expected = self.df.join(self.other_series)
        pd.testing.assert_frame_equal(expected, actual)

    def test_join_series_column(self):
        join = Join(0)
        actual = join(self.df, self.other_series)
        expected = self.df.join(self.other_series, 0)
        pd.testing.assert_frame_equal(expected, actual)

    def test_join_both_index(self):
        join = Join()
        actual = join(self.df, [self.other_df, self.other_series])
        expected = self.df.join([self.other_df, self.other_series])
        pd.testing.assert_frame_equal(expected, actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        join = Join(1, 2, three=3, four=4)
        expected = 'Join(1, 2, three=3, four=4)'
        self.assertEqual(expected, repr(join))

    def test_pickle_works(self):
        join = Join(1, 2, three=3, four=4)
        _ = pickle.loads(pickle.dumps(join))


if __name__ == '__main__':
    unittest.main()

import pickle
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
import unittest
from unittest.mock import Mock
from swak.pd import FrameGroupBy


class TestAttributeUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = Mock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])

    def test_has_args(self):
        groupby = FrameGroupBy()
        self.assertTrue(hasattr(groupby, 'args'))

    def test_has_kwargs(self):
        groupby = FrameGroupBy()
        self.assertTrue(hasattr(groupby, 'kwargs'))

    def test_args_empty(self):
        groupby = FrameGroupBy()
        self.assertTupleEqual((), groupby.args)

    def test_kwargs_empty(self):
        groupby = FrameGroupBy()
        self.assertDictEqual({}, groupby.kwargs)

    def test_args(self):
        groupby = FrameGroupBy(1, 2, three=3, four=4)
        self.assertTupleEqual((1, 2), groupby.args)

    def test_kwargs(self):
        groupby = FrameGroupBy(1, 2, three=3, four=4)
        self.assertDictEqual({'three': 3, 'four': 4}, groupby.kwargs)

    def test_callable(self):
        groupby = FrameGroupBy()
        self.assertTrue(callable(groupby))

    def test_called_with_args_kwargs(self):
        groupby = FrameGroupBy(1, 2, three=3, four=4)
        _ = groupby(self.mock)
        self.mock.groupby.assert_called_once_with(1, 2, three=3, four=4)

    def test_return_type_one_column(self):
        groupby = FrameGroupBy(0)
        actual = groupby(self.df)
        self.assertIsInstance(actual, DataFrameGroupBy)

    def test_return_type_two_columns(self):
        groupby = FrameGroupBy([0, 1])
        actual = groupby(self.df)
        self.assertIsInstance(actual, DataFrameGroupBy)


class TestMisc(unittest.TestCase):

    def test_repr_non_str(self):
        groupby = FrameGroupBy(1, 2, three=3, four=4)
        expected = 'FrameGroupBy(1, 2, three=3, four=4)'
        self.assertEqual(expected, repr(groupby))

    def test_repr_str(self):
        groupby = FrameGroupBy('1', '2', three='3', four='4')
        expected = "FrameGroupBy('1', '2', three='3', four='4')"
        self.assertEqual(expected, repr(groupby))

    def test_pickle_works(self):
        groupby = FrameGroupBy(1, 2, three=3, four=4)
        _ = pickle.loads(pickle.dumps(groupby))


if __name__ == '__main__':
    unittest.main()

import pickle
import pandas as pd
import unittest
from unittest.mock import Mock
from swak.pd import FrameGroupByAgg


class TestAttributeUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.mock = Mock()
        self.df = pd.DataFrame([[1, 5], [2, 6], [3, 7], [4, 8]])

    def test_has_args(self):
        agg = FrameGroupByAgg()
        self.assertTrue(hasattr(agg, 'args'))

    def test_has_kwargs(self):
        agg = FrameGroupByAgg()
        self.assertTrue(hasattr(agg, 'kwargs'))

    def test_args_empty(self):
        agg = FrameGroupByAgg()
        self.assertTupleEqual((), agg.args)

    def test_kwargs_empty(self):
        agg = FrameGroupByAgg()
        self.assertDictEqual({}, agg.kwargs)

    def test_args(self):
        agg = FrameGroupByAgg(1, 2, three=3, four=4)
        self.assertTupleEqual((1, 2), agg.args)

    def test_kwargs(self):
        agg = FrameGroupByAgg(1, 2, three=3, four=4)
        self.assertDictEqual({'three': 3, 'four': 4}, agg.kwargs)

    def test_callable(self):
        agg = FrameGroupByAgg()
        self.assertTrue(callable(agg))

    def test_called_with_args_kwargs(self):
        agg = FrameGroupByAgg(1, 2, three=3, four=4)
        _ = agg(self.mock)
        self.mock.agg.assert_called_once_with(1, 2, three=3, four=4)

    def test_return_type_one_agg(self):
        agg = FrameGroupByAgg('min')
        actual = agg(self.df.groupby(0))
        self.assertIsInstance(actual, pd.DataFrame)

    def test_return_type_two_aggs(self):
        agg = FrameGroupByAgg(['min', 'max'])
        actual = agg(self.df.groupby([0]))
        self.assertIsInstance(actual, pd.DataFrame)


class TestMisc(unittest.TestCase):

    def test_repr_non_str(self):
        agg = FrameGroupByAgg(1, 2, three=3, four=4)
        expected = 'FrameGroupByAgg(1, 2, three=3, four=4)'
        self.assertEqual(expected, repr(agg))

    def test_repr_str(self):
        agg = FrameGroupByAgg('1', '2', three='3', four='4')
        expected = "FrameGroupByAgg('1', '2', three='3', four='4')"
        self.assertEqual(expected, repr(agg))

    def test_pickle_works(self):
        agg = FrameGroupByAgg(1, 2, three=3, four=4)
        _ = pickle.loads(pickle.dumps(agg))


if __name__ == '__main__':
    unittest.main()

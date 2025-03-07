import pickle
import unittest
from unittest.mock import Mock
from swak.pd import Assign


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        assign = Assign()
        self.assertTrue(hasattr(assign, 'cols'))
        self.assertDictEqual({}, assign.cols)

    def test_arg(self):
        expected = {'one': 1, 'two': 2}
        assign = Assign(expected)
        self.assertDictEqual(expected, assign.cols)

    def test_kwargs(self):
        expected = {'one': 1, 'two': 2}
        assign = Assign(**expected)
        self.assertDictEqual(expected, assign.cols)

    def test_arg_and_kwargs(self):
        expected = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
        assign = Assign({'one': 1, 'two': 2}, three=3, four=4)
        self.assertDictEqual(expected, assign.cols)

    def test_kwargs_trump_arg(self):
        expected = {'one': 1, 'two': 3, 'three': 4}
        assign = Assign({'one': 1, 'two': 2}, two=3, three=4)
        self.assertDictEqual(expected, assign.cols)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        assign = Assign()
        self.assertTrue(callable(assign))

    def test_assign_called_with_cols(self):
        df = Mock()
        assign = Assign({'one': 1, 'two': 2, 'three': 3, 'four': 4})
        _ = assign(df)
        df.assign.assert_called_once_with(**assign.cols)


class TestMisc(unittest.TestCase):

    def test_repr_arg(self):
        assign = Assign({'one': 1, 'two': 2})
        expected = "Assign(one=1, two=2)"
        self.assertEqual(expected, repr(assign))

    def test_repr_kwargs(self):
        assign = Assign(one=1, two=2)
        expected = "Assign(one=1, two=2)"
        self.assertEqual(expected, repr(assign))

    def test_pickle_works(self):
        assign = Assign({'one': 1, 'two': 2})
        _ = pickle.loads(pickle.dumps(assign))

if __name__ == '__main__':
    unittest.main()

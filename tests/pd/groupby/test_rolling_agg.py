import pickle
import unittest
from unittest.mock import Mock
from swak.pd import RollingGroupByAgg


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.agg = RollingGroupByAgg(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.agg, 'func'))

    def test_func(self):
        self.assertIs(self.agg.func, f)

    def test_has_args(self):
        self.assertTrue(hasattr(self.agg, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.agg.args)

    def test_has_kwargs(self):
        self.assertTrue(self.agg, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.agg.kwargs)



class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.agg = RollingGroupByAgg(f, *self.args, **self.kwargs)

    def test_args(self):
        self.assertTupleEqual(self.args, self.agg.args)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.agg.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.agg = RollingGroupByAgg(f, *self.args, **self.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.agg))

    def test_rolling_called(self):
        grouped = Mock()
        _ = self.agg(grouped)
        grouped.agg.assert_called_once_with(f, *self.args, **self.kwargs)

    def test_return_value(self):
        grouped = Mock()
        grouped.agg = Mock(return_value='cheese')
        actual = self.agg(grouped)
        self.assertEqual('cheese', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        agg = RollingGroupByAgg(f)
        expected = 'RollingGroupByAgg(f)'
        self.assertEqual(expected, repr(agg))

    def test_custom_repr(self):
        agg = RollingGroupByAgg(f, 'foo', 1, answer=42)
        expected = "RollingGroupByAgg(f, 'foo', 1, answer=42)"
        self.assertEqual(expected, repr(agg))

    def test_pickle_works_with_function(self):
        agg = RollingGroupByAgg(f, 'foo', 1, answer=42)
        _ = pickle.loads(pickle.dumps(agg))

    def test_pickle_raises_with_lambda(self):
        agg = RollingGroupByAgg(lambda x: x.mean(), 'foo', 1, answer=42)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(agg))


if __name__ == '__main__':
    unittest.main()

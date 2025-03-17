import pickle
import unittest
from unittest.mock import Mock
from swak.pd import RollingGroupBy


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.roll = RollingGroupBy()

    def test_has_args(self):
        self.assertTrue(hasattr(self.roll, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.roll.args)

    def test_has_kwargs(self):
        self.assertTrue(self.roll, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.roll.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.roll = RollingGroupBy(*self.args, **self.kwargs)

    def test_args(self):
        self.assertTupleEqual(self.args, self.roll.args)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.roll.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.roll = RollingGroupBy(*self.args, **self.kwargs)

    def test_callable(self):
        self.assertTrue(callable(self.roll))

    def test_rolling_called(self):
        df = Mock()
        _ = self.roll(df)
        df.rolling.assert_called_once_with(*self.args, **self.kwargs)

    def test_return_value(self):
        df = Mock()
        df.rolling = Mock(return_value='cheese')
        actual = self.roll(df)
        self.assertEqual('cheese', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        roll = RollingGroupBy()
        expected = 'RollingGroupBy()'
        self.assertEqual(expected, repr(roll))

    def test_custom_repr(self):
        roll = RollingGroupBy('foo', 1, answer=42)
        expected = "RollingGroupBy('foo', 1, answer=42)"
        self.assertEqual(expected, repr(roll))

    def test_pickle_works(self):
        roll = RollingGroupBy('foo', 1, answer=42)
        _ = pickle.loads(pickle.dumps(roll))


if __name__ == '__main__':
    unittest.main()

import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Mapper


def f(x):
    return x.mean()

class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.map = Mapper(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.map, 'func'))

    def test_func(self):
        self.assertIs(self.map.func, f)

    def test_has_na_action(self):
        self.assertTrue(hasattr(self.map, 'na_action'))

    def test_na_action(self):
        self.assertIsNone(self.map.na_action)

    def test_has_engine(self):
        self.assertTrue(hasattr(self.map, 'engine'))

    def test_engine(self):
        self.assertIsNone(self.map.engine)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.map, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.map.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.func = {'foo': 'bar'}
        self.na_action = 'ignore'
        self.engine = 'numba'
        self.kwargs = {'answer': 42}
        self.map = Mapper(
            self.func,
            self.na_action,
            self.engine,
            **self.kwargs
        )

    def test_func(self):
        self.assertDictEqual(self.func, self.map.func)

    def test_na_action(self):
        self.assertEqual(self.na_action, self.map.na_action)

    def test_engine(self):
        self.assertEqual(self.engine, self.map.engine)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.map.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.func = {'foo': 'bar'}
        self.na_action = 'ignore'
        self.engine = 'numba'
        self.kwargs = {'answer': 42}
        self.map = Mapper(
            self.func,
            self.na_action,
            self.engine,
            **self.kwargs
        )
        self.default = Mapper(f)

    def test_dataframe_default(self):
        df = pd.DataFrame(range(10))
        df.map = Mock(return_value='answer')
        actual = self.default(df)
        df.map.assert_called_once_with(f, None)
        self.assertEqual('answer', actual)

    def test_dataframe_custom(self):
        df = pd.DataFrame(range(10))
        df.map = Mock(return_value='answer')
        actual = self.map(df)
        df.map.assert_called_once_with(
            self.func,
            self.na_action,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_series_default(self):
        df = pd.Series(range(10))
        df.map = Mock(return_value='answer')
        actual = self.default(df)
        df.map.assert_called_once_with(f, None, None)
        self.assertEqual('answer', actual)

    def test_series_custom(self):
        df = pd.Series(range(10))
        df.map = Mock(return_value='answer')
        actual = self.map(df)
        df.map.assert_called_once_with(
            self.func,
            self.na_action,
            self.engine,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_raises_on_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.map(2)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.func = {'foo': 'bar'}
        self.na_action = 'ignore'
        self.engine = 'numba'
        self.kwargs = {'answer': 42}
        self.map = Mapper(
            self.func,
            self.na_action,
            self.engine,
            **self.kwargs
        )
        self.default = Mapper(f)

    def test_default_repr(self):
        expected = 'Mapper(f, None, None)'
        self.assertEqual(expected, repr(self.default))

    def test_custom_repr(self):
        expected = "Mapper(dict, 'ignore', 'numba', answer=42)"
        self.assertEqual(expected, repr(self.map))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.map))


if __name__ == '__main__':
    unittest.main()

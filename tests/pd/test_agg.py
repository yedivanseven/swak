import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Agg


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.agg = Agg()

    def test_has_func(self):
        self.assertTrue(hasattr(self.agg, 'func'))

    def test_func(self):
        self.assertIsNone(self.agg.func)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.agg, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.agg.axis, int)
        self.assertEqual(0, self.agg.axis)

    def test_has_args(self):
        self.assertTrue(hasattr(self.agg, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.agg.args)

    def test_has_engine(self):
        self.assertTrue(hasattr(self.agg, 'engine'))

    def test_engine(self):
        self.assertIsNone(self.agg.engine)

    def test_has_engine_kwargs(self):
        self.assertTrue(hasattr(self.agg, 'engine_kwargs'))

    def test_engine_kwargs(self):
        self.assertIsNone(self.agg.engine_kwargs)

    def test_has_kwargs(self):
        self.assertTrue(self.agg, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.agg.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.axis = 'columns'
        self.args = 'foo', 1
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'bar': 2, 'answer': 42}
        self.agg = Agg(
            f,
            self.axis,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )

    def test_func(self):
        self.assertIs(self.agg.func, f)

    def test_axis(self):
        self.assertEqual(self.axis, self.agg.axis)

    def test_args(self):
        self.assertTupleEqual(self.args, self.agg.args)

    def test_engine(self):
        self.assertEqual(self.engine, self.agg.engine)

    def test_engine_kwargs(self):
        self.assertDictEqual(self.engine_kwargs, self.agg.engine_kwargs)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.agg.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.axis = 'columns'
        self.args = 'foo', 1
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'bar': 2, 'answer': 42}
        self.agg = Agg(
            f,
            self.axis,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
    def test_callable(self):
        self.assertTrue(callable(self.agg))

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            self.axis,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_dataframe_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0)
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            0,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_series_groupby(self):
        df = pd.Series([1, 2, 3, 4]).groupby([1, 1, 2, 2])
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_rolling(self):
        df = pd.DataFrame(range(10)).rolling(2)
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('answer', actual)

    def test_rolling_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0).rolling(2)
        df.agg = Mock(return_value='answer')
        actual = self.agg(df)
        df.agg.assert_called_once_with(
            f,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('answer', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        agg = Agg()
        expected = 'Agg(None, 0, engine=None, engine_kwargs=None)'
        self.assertEqual(expected, repr(agg))

    def test_custom_repr(self):
        agg = Agg(
            f,
            1,
            'foo',
            engine='numba',
            engine_kwargs={'nogil': True},
            answer=42
        )
        expected = ("Agg(f, 1, 'foo', engine='numba', "
                    "engine_kwargs={'nogil': True}, answer=42)")
        self.assertEqual(expected, repr(agg))

    def test_pickle_works_with_function(self):
        agg = Agg(f)
        _ = pickle.loads(pickle.dumps(agg))

    def test_pickle_raises_with_lambda(self):
        agg = Agg(lambda x: x.mean())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(agg))


if __name__ == '__main__':
    unittest.main()

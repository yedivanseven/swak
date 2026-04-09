import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Transform


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.transform = Transform(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.transform, 'func'))

    def test_func(self):
        self.assertIs(self.transform.func, f)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.transform, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.transform.axis, int)
        self.assertEqual(0, self.transform.axis)

    def test_has_args(self):
        self.assertTrue(hasattr(self.transform, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.transform.args)

    def test_has_engine(self):
        self.assertTrue(hasattr(self.transform, 'engine'))

    def test_engine(self):
        self.assertIsNone(self.transform.engine)

    def test_has_engine_kwargs(self):
        self.assertTrue(hasattr(self.transform, 'engine_kwargs'))

    def test_engine_kwargs(self):
        self.assertIsNone(self.transform.engine_kwargs)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.transform, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.transform.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.axis = 'rows'
        self.args = 'foo', 'bar'
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'answer': 42}
        self.transform = Transform(
            f,
            self.axis,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )

    def test_axis(self):
        self.assertEqual(self.axis, self.transform.axis)

    def test_args(self):
        self.assertTupleEqual(self.args, self.transform.args)

    def test_engine(self):
        self.assertEqual(self.engine, self.transform.engine)

    def test_engine_kwargs(self):
        self.assertDictEqual(self.engine_kwargs, self.transform.engine_kwargs)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.transform.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.axis = 'rows'
        self.args = 'foo', 'bar'
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'answer': 42}
        self.transform = Transform(
            f,
            self.axis,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.transform = Mock(return_value='expected')
        actual = self.transform(df)
        df.transform.assert_called_once_with(
            f,
            self.axis,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_dataframe_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0)
        df.transform = Mock(return_value='expected')
        actual = self.transform(df)
        df.transform.assert_called_once_with(
            f,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_series(self):
        df = pd.Series(range(10))
        df.transform = Mock(return_value='expected')
        actual = self.transform(df)
        df.transform.assert_called_once_with(
            f,
            0,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_series_groupby(self):
        df = pd.Series(range(4)).groupby([1, 1, 2, 2])
        df.transform = Mock(return_value='expected')
        actual = self.transform(df)
        df.transform.assert_called_once_with(
            f,
            *self.args,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_resampler(self):
        df = pd.DataFrame(
            range(10),
            index=pd.bdate_range('2012-01-01', periods=10)
        ).resample('1D')
        df.transform = Mock(return_value='expected')
        actual = self.transform(df)
        df.transform.assert_called_once_with(
            f,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('expected', actual)

    def test_raises_on_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.transform(2)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        transform = Transform(f)
        expected = 'Transform(f, 0, engine=None, engine_kwargs=None)'
        self.assertEqual(expected, repr(transform))

    def test_custom_repr(self):
        transform = Transform(
            f,
            'rows',
            'foo', 'bar',
            engine='numba',
            engine_kwargs={'nogil': True},
            answer=42
        )
        expected = ("Transform(f, 'rows', 'foo', 'bar', engine='numba', "
                    "engine_kwargs={'nogil': True}, answer=42)")
        self.assertEqual(expected, repr(transform))

    def test_pickle_works_with_function(self):
        transform = Transform(f)
        _ = pickle.loads(pickle.dumps(transform))

    def test_pickle_raises_with_lambda(self):
        transform = Transform(lambda x: x.mean())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(transform))


if __name__ == '__main__':
    unittest.main()

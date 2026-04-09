import pickle
import unittest
from unittest.mock import Mock
import pandas as pd
from swak.pd import Apply


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.apply = Apply(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.apply, 'func'))

    def test_func(self):
        self.assertIs(self.apply.func, f)

    def test_has_axis(self):
        self.assertTrue(hasattr(self.apply, 'axis'))

    def test_axis(self):
        self.assertIsInstance(self.apply.axis, int)
        self.assertEqual(0, self.apply.axis)

    def test_has_raw(self):
        self.assertTrue(hasattr(self.apply, 'raw'))

    def test_raw(self):
        self.assertIsInstance(self.apply.raw, bool)
        self.assertFalse(self.apply.raw)

    def test_has_result_type(self):
        self.assertTrue(hasattr(self.apply, 'result_type'))

    def test_result_type(self):
        self.assertIsNone(self.apply.result_type)

    def test_has_args(self):
        self.assertTrue(hasattr(self.apply, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.apply.args)

    def test_has_by_row(self):
        self.assertTrue(hasattr(self.apply, 'by_row'))

    def test_by_row(self):
        self.assertEqual('compat', self.apply.by_row)

    def test_has_engine(self):
        self.assertTrue(hasattr(self.apply, 'engine'))

    def test_engine(self):
        self.assertIsNone(self.apply.engine)

    def test_has_engine_kwargs(self):
        self.assertTrue(hasattr(self.apply, 'engine_kwargs'))

    def test_engine_kwargs(self):
        self.assertIsNone(self.apply.engine_kwargs)

    def test_has_kwargs(self):
        self.assertTrue(self.apply, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.apply.kwargs)


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.func = 'func'
        self.axis = 1
        self.raw = True
        self.result_type = 'expand'
        self.args = 'foo', 'bar'
        self.by_row = False
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'answer': 42}
        self.apply = Apply(
            self.func,
            self.axis,
            self.raw,
            self.result_type ,
            self.args,
            self.by_row,
            self.engine,
            self.engine_kwargs,
            **self.kwargs,
        )

    def test_func(self):
        self.assertEqual(self.func, self.apply.func)

    def test_axis(self):
        self.assertEqual(self.axis, self.apply.axis)

    def test_raw(self):
        self.assertEqual(self.raw, self.apply.raw)

    def test_result_type(self):
        self.assertEqual(self.result_type, self.apply.result_type)

    def test_args(self):
        self.assertTupleEqual(self.args, self.apply.args)

    def test_by_row(self):
        self.assertEqual(self.by_row, self.apply.by_row)

    def test_engine(self):
        self.assertEqual(self.engine, self.apply.engine)

    def test_has_engine_kwargs(self):
        self.assertTrue(hasattr(self.apply, 'engine_kwargs'))

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.apply.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.func = f
        self.axis = 1
        self.raw = True
        self.result_type = 'expand'
        self.args = 'foo', 'bar'
        self.by_row = False
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.kwargs = {'answer': 42}
        self.apply = Apply(
            self.func,
            self.axis,
            self.raw,
            self.result_type ,
            self.args,
            self.by_row,
            self.engine,
            self.engine_kwargs,
            **self.kwargs,
        )

    def test_dataframe(self):
        df = pd.DataFrame(range(10))
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            axis=self.axis,
            raw=self.raw,
            result_type=self.result_type,
            args=self.args,
            by_row=self.by_row,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            **self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_series(self):
        df = pd.DataFrame(range(10))[0]
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            args=self.args,
            by_row=self.by_row,
            **self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_rolling(self):
        df = pd.DataFrame(range(10)).rolling(1)
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            raw=self.raw,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            args=self.args,
            kwargs=self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_resample(self):
        df = pd.DataFrame(
            range(10),
            index=pd.bdate_range('2012-01-01', periods=10)
        ).resample('1D')
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_dataframe_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0)
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            *self.args,
            include_groups=False,
            **self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_series_groupby(self):
        df = pd.DataFrame(range(4))[0].groupby([1, 1, 2, 2])
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            *self.args,
            **self.kwargs
        )
        self.assertEqual('return_value', actual)

    def test_rolling_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0).rolling(1)
        df.apply = Mock(return_value='return_value')
        actual = self.apply(df)
        df.apply.assert_called_once_with(
            self.func,
            raw=self.raw,
            engine=self.engine,
            engine_kwargs=self.engine_kwargs,
            args=self.args,
            kwargs=self.kwargs
        )
        self.assertEqual('return_value', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        agg = Apply(f)
        expected = ("Apply(f, axis=0, raw=False, result_type=None, args=(), "
                    "by_row='compat', engine=None, engine_kwargs=None)")
        self.assertEqual(expected, repr(agg))

    def test_custom_repr(self):
        agg = Apply(
            f,
            axis=1,
            raw=True,
            result_type='expand',
            args=('foo', 'bar'),
            by_row=False,
            engine='numba',
            engine_kwargs={'nogil': True},
            answer=42
        )
        expected = ("Apply(f, axis=1, raw=True, result_type='expand', "
                    "args=('foo', 'bar'), by_row=False, engine='numba', "
                    "engine_kwargs={'nogil': True}, answer=42)")
        self.assertEqual(expected, repr(agg))

    def test_pickle_works_with_function(self):
        agg = Apply(f)
        _ = pickle.loads(pickle.dumps(agg))

    def test_pickle_raises_with_lambda(self):
        agg = Apply(lambda x: x.mean())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(agg))


if __name__ == '__main__':
    unittest.main()

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
        self.assertEqual('python', self.apply.engine)

    def test_has_engine_kwargs(self):
        self.assertTrue(hasattr(self.apply, 'engine_kwargs'))

    def test_engine_kwargs(self):
        self.assertIsNone(self.apply.engine_kwargs)

    def test_has_include_groups(self):
        self.assertTrue(hasattr(self.apply, 'include_groups'))

    def test_include_groups(self):
        self.assertIsInstance(self.apply.include_groups, bool)
        self.assertTrue(self.apply.include_groups)

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
        self.include_groups = False
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
            self.include_groups,
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

    def test_include_groups(self):
        self.assertEqual(self.include_groups, self.apply.include_groups)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.apply.kwargs)


class TestCallSignature(unittest.TestCase):

    def setUp(self):
        self.func = f
        self.axis = 1
        self.raw = True
        self.result_type = 'expand'
        self.args = 'foo', 'bar'
        self.by_row = False
        self.engine = 'numba'
        self.engine_kwargs = {'nogil': True}
        self.include_groups = False
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
            self.include_groups,
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
            self.args,
            by_row=self.by_row,
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
            include_groups=self.include_groups,
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

    def test_expanding_groupby(self):
        df = pd.DataFrame(range(10)).groupby(0).expanding(1)
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

    def test_raises_on_wrong_type(self):
        with self.assertRaises(TypeError):
            _ = self.apply(2)


class TestUsage(unittest.TestCase):

    def test_dataframe_function(self):
        df = pd.DataFrame(range(10))
        apply = Apply(
            lambda x, y: 2*x + y,
            axis=1,
            raw=True,
            result_type='broadcast',
            args=(3,),
            by_row=False,
            engine='engine',
            engine_kwargs={'answer': 42}
        )
        actual = apply(df)
        pd.testing.assert_frame_equal(actual, df*2 + 3)

    def test_dataframe_list(self):
        df = pd.DataFrame(range(10))
        apply = Apply(
            ['mean'],
            axis=1,
            raw=True,
            result_type='broadcast',
            args=(3,),
            by_row=False,
            engine='engine',
            engine_kwargs={'answer': 42}
        )
        actual = apply(df)
        expected = pd.DataFrame(range(10), columns=['mean'], dtype=float)
        pd.testing.assert_frame_equal(actual, expected)

    def test_dataframe_str(self):
        df = pd.DataFrame(range(10))
        apply = Apply(
            'mean',
            axis=1,
            result_type='expand'
        )
        actual = apply(df)
        expected = pd.Series(range(10), dtype=float)
        pd.testing.assert_series_equal(actual, expected)

    def test_dataframe_map(self):
        df = pd.DataFrame(range(10))
        apply = Apply(
            {1: 'mean'},
            axis=1,
            result_type='expand'
        )
        actual = apply(df)
        expected = pd.Series({1: 1.0})
        pd.testing.assert_series_equal(actual, expected)

    def test_series_function(self):
        df = pd.Series(range(10))
        apply = Apply(
            lambda x, y: 2 * x + y,
            axis=1,
            raw=True,
            result_type='broadcast',
            args=(3,),
            by_row=False,
            engine='dss',
            engine_kwargs={'answer': 42}
        )
        actual = apply(df)
        pd.testing.assert_series_equal(actual, df * 2 + 3)

    def test_series_list(self):
        df = pd.DataFrame(range(10))
        apply = Apply(
            ['mean'],
            axis=1,
            raw=True,
            result_type='broadcast',
            args=(3,),
            by_row=False,
            engine='dss',
            engine_kwargs={'answer': 42}
        )
        actual = apply(df)
        expected = pd.DataFrame(range(10), columns=['mean'], dtype=float)
        pd.testing.assert_frame_equal(actual, expected)

    def test_series_str(self):
        df = pd.Series(range(10))
        apply = Apply(
            'mean',
            axis=1,
            raw=False,
            result_type='expand'
        )
        actual = apply(df)
        self.assertEqual(4.5, actual)

    def test_series_map(self):
        df = pd.Series(range(10))
        apply = Apply(
            {1: 'mean'},
            axis=1,
            raw=False,
            result_type='expand'
        )
        actual = apply(df)
        expected = pd.Series({1: 4.5})
        pd.testing.assert_series_equal(actual, expected)

    # ToDo: COntinue with other pandas objects


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        agg = Apply(f)
        expected = ("Apply(f, axis=0, raw=False, result_type=None, args=(), "
                    "by_row='compat', engine='python', engine_kwargs=None, "
                    "include_groups=True)")
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
            include_groups=False,
            answer=42
        )
        expected = ("Apply(f, axis=1, raw=True, result_type='expand', "
                    "args=('foo', 'bar'), by_row=False, engine='numba', "
                    "engine_kwargs={'nogil': True}, include_groups=False, "
                    "answer=42)")
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

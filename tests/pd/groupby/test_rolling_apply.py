import pickle
import unittest
from unittest.mock import Mock
from swak.pd import RollingGroupByApply


def f(x):
    return x.mean()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.apply = RollingGroupByApply(f)

    def test_has_func(self):
        self.assertTrue(hasattr(self.apply, 'func'))

    def test_func(self):
        self.assertIs(self.apply.func, f)

    def test_has_args(self):
        self.assertTrue(hasattr(self.apply, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.apply.args)

    def test_has_raw(self):
        self.assertTrue(hasattr(self.apply, 'raw'))

    def test_raw(self):
        self.assertIsInstance(self.apply.raw, bool)
        self.assertFalse(self.apply.raw)

    def test_has_engine(self):
        self.assertTrue(hasattr(self.apply, 'engine'))

    def test_engine(self):
        self.assertIsNone(self.apply.engine)

    def test_has_engine_kws(self):
        self.assertTrue(hasattr(self.apply, 'engine_kws'))

    def test_engine_kws(self):
        self.assertIsNone(self.apply.engine_kws)

    def test_has_kwargs(self):
        self.assertTrue(self.apply, 'kwargs')

    def test_kwargs(self):
        self.assertDictEqual({}, self.apply.kwargs)



class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.raw = True
        self.engine = 'cython'
        self.engine_kws = {'baz': False}
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.apply = RollingGroupByApply(
            f,
            self.raw,
            self.engine,
            self.engine_kws,
            *self.args, **self.kwargs
        )

    def test_raw(self):
        self.assertIs(self.apply.raw, self.raw)

    def test_engine(self):
        self.assertEqual(self.engine, self.apply.engine)

    def test_engine_kws(self):
        self.assertDictEqual(self.engine_kws, self.apply.engine_kws)

    def test_args(self):
        self.assertTupleEqual(self.args, self.apply.args)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.apply.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.raw = True
        self.engine = 'cython'
        self.engine_kws = {'baz': False}
        self.args = 'foo', 1
        self.kwargs = {'bar': 2, 'answer': 42}
        self.apply = RollingGroupByApply(
            f,
            self.raw,
            self.engine,
            self.engine_kws,
            *self.args, **self.kwargs
        )


    def test_callable(self):
            self.assertTrue(callable(self.apply))

    def test_rolling_called(self):
        grouped = Mock()
        _ = self.apply(grouped)
        grouped.apply.assert_called_once_with(
            f,
            self.raw,
            self.engine,
            self.engine_kws,
            self.args,
            self.kwargs
        )

    def test_return_value(self):
        grouped = Mock()
        grouped.apply = Mock(return_value='cheese')
        actual = self.apply(grouped)
        self.assertEqual('cheese', actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        apply = RollingGroupByApply(f)
        expected = 'RollingGroupByApply(f, False, None, None)'
        self.assertEqual(expected, repr(apply))

    def test_custom_repr(self):
        apply = RollingGroupByApply(
            f,
            True,
            'cython',
            {'baz': False},
            1,
            answer=42
        )
        expected = ("RollingGroupByApply(f, True, 'cython', "
                    "{'baz': False}, 1, answer=42)")
        self.assertEqual(expected, repr(apply))

    def test_pickle_works_with_function(self):
        apply = RollingGroupByApply(f)
        _ = pickle.loads(pickle.dumps(apply))

    def test_pickle_raises_with_lambda(self):
        apply = RollingGroupByApply(lambda x: x.mean())
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(apply))


if __name__ == '__main__':
    unittest.main()

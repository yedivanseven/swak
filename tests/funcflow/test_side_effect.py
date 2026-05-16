import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import SideEffect
from swak.funcflow.exceptions import SideEffectError


def f(_) -> None:
    pass


def g(x: int) -> None:
    _ = 1 / x


class TestAttributed(unittest.TestCase):

    def setUp(self):
        self.side = SideEffect(f)

    def test_has_call(self):
        self.assertTrue(hasattr(self.side, 'call'))

    def test_call(self):
        self.assertIs(self.side.call, f)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        side = SideEffect(f)
        self.assertTrue(callable(side))

    def test_call(self):
        call = Mock()
        side = SideEffect(call)
        _ = side('foo', 'bar', 42)
        call.assert_called_once_with('foo', 'bar', 42)

    def test_return_value_no_args(self):
        call = Mock()
        side = SideEffect(call)
        actual = side()
        self.assertTupleEqual((), actual)

    def test_return_value_one_arg(self):
        call = Mock()
        side = SideEffect(call)
        obj = object()
        actual = side(obj)
        self.assertIs(actual, obj)

    def test_return_value_multiple_arg(self):
        call = Mock()
        side = SideEffect(call)
        actual = side('foo', 'bar', 42)
        self.assertTupleEqual(('foo', 'bar', 42), actual)

    def test_raises(self):
        side = SideEffect(g)
        with self.assertRaises(SideEffectError):
            _ = side(0)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        side = SideEffect(f)
        self.assertEqual('SideEffect(f)', repr(side))

    def test_pickle_works(self):
        side = SideEffect(f)
        _ = pickle.loads(pickle.dumps(side))

    def test_pickle_raises_with_lambda(self):
        side = SideEffect(lambda x: None)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(side))


if __name__ == '__main__':
    unittest.main()

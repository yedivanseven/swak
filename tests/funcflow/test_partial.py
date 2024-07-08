import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Partial


def f(x, y):
    return x + y


class TestInstantiation(unittest.TestCase):

    def test_call_only(self):
        _ = Partial(f)

    def test_args(self):
        _ = Partial(f, 42, 'foo', 1.23)

    def test_kwargs(self):
        _ = Partial(f, answer=42, bar='foo')

    def test_args_and_kwargs(self):
        _ = Partial(f, 5, 'baz', 6.7, answer=42, bar='foo')

    def test_type_args_and_kwargs(self):
        _ = Partial[int](f, 5, 'baz', 6.7, answer=42, bar='foo')

    def test_callable(self):
        partial = Partial(f, 'foo', answer=42)
        self.assertTrue(callable(partial))


class TestCall(unittest.TestCase):

    def setUp(self) -> None:
        self.call = Mock()

    def test_call_called(self):
        partial = Partial(self.call)
        _ = partial()
        self.call.assert_called_once()
        self.call.assert_called_once_with()

    def test_call_called_with_instance_args(self):
        partial = Partial(self.call, 42, 'foo', 1.23)
        _ = partial()
        self.call.assert_called_once()
        self.call.assert_called_once_with(42, 'foo', 1.23)

    def test_call_called_with_instance_kwargs(self):
        partial = Partial(self.call, answer=42, bar='foo', susi=1.23)
        _ = partial()
        self.call.assert_called_once()
        self.call.assert_called_once_with(answer=42, bar='foo', susi=1.23)

    def test_call_called_with_instance_args_and_kwargs(self):
        args = 5, 'baz', 6.7
        kwargs = {'answer': 42, 'bar': 'foo', 'susi': 1.23}
        partial = Partial(self.call, *args, **kwargs)
        _ = partial()
        self.call.assert_called_once()
        self.call.assert_called_once_with(*args, **kwargs)

    def test_call_called_with_call_args(self):
        partial = Partial(self.call)
        _ = partial(42, 'foo', 1.23)
        self.call.assert_called_once()
        self.call.assert_called_once_with(42, 'foo', 1.23)

    def test_call_called_with_call_kwargs(self):
        partial = Partial(self.call)
        _ = partial(answer=42, bar='foo', susi=1.23)
        self.call.assert_called_once()
        self.call.assert_called_once_with(answer=42, bar='foo', susi=1.23)

    def test_call_called_with_call_args_and_kwargs(self):
        args = 5, 'baz', 6.7
        kwargs = {'answer': 42, 'bar': 'foo', 'susi': 1.23}
        partial = Partial(self.call)
        _ = partial(*args, **kwargs)
        self.call.assert_called_once()
        self.call.assert_called_once_with(*args, **kwargs)

    def test_call_arg_ordering(self):
        partial = Partial(self.call, 4, 'foo')
        _ = partial(2, 'bar')
        self.call.assert_called_once()
        self.call.assert_called_once_with(4, 'foo', 2, 'bar')

    def test_call_kwarg_adding(self):
        partial = Partial(self.call, foo='bar')
        _ = partial(answer=42)
        self.call.assert_called_once()
        self.call.assert_called_once_with(foo='bar', answer=42)

    def test_call_kwarg_overwriting(self):
        partial = Partial(self.call, foo='bar', answer=49)
        _ = partial(answer=42)
        self.call.assert_called_once()
        self.call.assert_called_once_with(foo='bar', answer=42)

    def test_return_value(self):
        partial = Partial(f, 1)
        actual = partial(2)
        self.assertIsInstance(actual, int)
        self.assertEqual(3, actual)


class TestMisc(unittest.TestCase):

    def test_pickle_works(self):
        partial = Partial(f, 'foo', answer=42)
        _ = pickle.dumps(partial)

    def test_pickle_raises_lambda(self):
        partial = Partial(lambda x: x, 'foo', answer=42)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(partial)

    def test_representation(self):
        partial = Partial(f, 'foo', answer=42)
        self.assertEqual("Partial(f, 'foo', answer=42)", repr(partial))

    def test_type_annotation(self):
        _ = Partial[int](f, 42)


if __name__ == '__main__':
    unittest.main()

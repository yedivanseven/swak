import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Partial
from swak.magic import ArgRepr, IndentRepr


def f(x, y):
    return x + y


class Cls:

    @classmethod
    def c(cls):
        pass

    def m(self):
        pass

    @staticmethod
    def s(x):
        pass


class Call:

    def __call__(self):
        pass


class A(ArgRepr):

    def __init__(self, *xs):
        super().__init__(*xs)

    def __call__(self):
        pass


class Ind(IndentRepr):

    def __init__(self, *xs):
        super().__init__(*xs)

    def __call__(self):
        pass


class TestAttributes(unittest.TestCase):

    def test_call_only(self):
        p = Partial(f)
        self.assertTrue(hasattr(p, 'call'))
        self.assertIs(f, p.call)
        self.assertTrue(hasattr(p, 'args'))
        self.assertTupleEqual((), p.args)
        self.assertTrue(hasattr(p, 'kwargs'))
        self.assertDictEqual({}, p.kwargs)

    def test_args(self):
        p = Partial(f, 42, 'foo', 1.23)
        self.assertTrue(hasattr(p, 'call'))
        self.assertIs(f, p.call)
        self.assertTrue(hasattr(p, 'args'))
        self.assertTupleEqual((42, 'foo', 1.23), p.args)
        self.assertTrue(hasattr(p, 'kwargs'))
        self.assertDictEqual({}, p.kwargs)

    def test_kwargs(self):
        p = Partial(f, answer=42, bar='foo')
        self.assertTrue(hasattr(p, 'call'))
        self.assertIs(f, p.call)
        self.assertTrue(hasattr(p, 'args'))
        self.assertTupleEqual((), p.args)
        self.assertTrue(hasattr(p, 'kwargs'))
        self.assertDictEqual({'answer': 42, 'bar': 'foo'}, p.kwargs)

    def test_args_and_kwargs(self):
        p = Partial(f, 5, 'baz', 6.7, answer=42, bar='foo')
        self.assertTrue(hasattr(p, 'call'))
        self.assertIs(f, p.call)
        self.assertTrue(hasattr(p, 'args'))
        self.assertTupleEqual((5, 'baz', 6.7), p.args)
        self.assertTrue(hasattr(p, 'kwargs'))
        self.assertDictEqual({'answer': 42, 'bar': 'foo'}, p.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.call = Mock()

    def test_callable(self):
        args = 5, 'baz', 6.7
        kwargs = {'answer': 42, 'bar': 'foo', 'susi': 1.23}
        partial = Partial(self.call, *args, **kwargs)
        self.assertTrue(callable(partial))

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

    def test_lambda_repr(self):
        partial = Partial(lambda x: x, 'foo', answer=42)
        self.assertEqual("Partial(lambda, 'foo', answer=42)", repr(partial))

    def test_function_repr(self):
        partial = Partial(f, 'foo', answer=42)
        self.assertEqual("Partial(f, 'foo', answer=42)", repr(partial))

    def test_class_repr(self):
        partial = Partial(Cls, 'foo', answer=42)
        self.assertEqual("Partial(Cls, 'foo', answer=42)", repr(partial))

    def test_obj_repr(self):
        partial = Partial(Call(), 'foo', answer=42)
        self.assertEqual("Partial(Call(...), 'foo', answer=42)", repr(partial))

    def test_classmethod_repr(self):
        partial = Partial(Cls.m, 'foo', answer=42)
        self.assertEqual("Partial(Cls.m, 'foo', answer=42)", repr(partial))

    def test_staticmethod_repr(self):
        partial = Partial(Cls().s, 'foo', answer=42)
        self.assertEqual("Partial(Cls.s, 'foo', answer=42)", repr(partial))

    def test_method_repr(self):
        partial = Partial(Cls().m, 'foo', answer=42)
        self.assertEqual("Partial(Cls.m, 'foo', answer=42)", repr(partial))

    def test_argrepr(self):
        partial = Partial(A(1), A(2), answer=42)
        self.assertEqual("Partial(A(1), A(2), answer=42)", repr(partial))

    def test_indentrepr(self):
        partial = Partial(Ind(1, 2, 3), Ind(1, 2, 3), answer=42)
        self.assertEqual("Partial(Ind[3], Ind[3], answer=42)", repr(partial))

    def test_type_annotation(self):
        _ = Partial[int](f, 42)


if __name__ == '__main__':
    unittest.main()

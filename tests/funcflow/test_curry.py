import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Curry
from swak.misc import ArgRepr, IndentRepr


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

    def __call__(self):
        pass


class TestAttributes(unittest.TestCase):

    def test_call_only(self):
        c = Curry(f)
        self.assertTrue(hasattr(c, 'call'))
        self.assertIs(f, c.call)
        self.assertTrue(hasattr(c, 'args'))
        self.assertTupleEqual((), c.args)
        self.assertTrue(hasattr(c, 'kwargs'))
        self.assertDictEqual({}, c.kwargs)

    def test_args(self):
        c = Curry(f, 42, 'foo', 1.23)
        self.assertTrue(hasattr(c, 'call'))
        self.assertIs(f, c.call)
        self.assertTrue(hasattr(c, 'args'))
        self.assertTupleEqual((42, 'foo', 1.23), c.args)
        self.assertTrue(hasattr(c, 'kwargs'))
        self.assertDictEqual({}, c.kwargs)

    def test_kwargs(self):
        c = Curry(f, answer=42, bar='foo')
        self.assertTrue(hasattr(c, 'call'))
        self.assertIs(f, c.call)
        self.assertTrue(hasattr(c, 'args'))
        self.assertTupleEqual((), c.args)
        self.assertTrue(hasattr(c, 'kwargs'))
        self.assertDictEqual({'answer': 42, 'bar': 'foo'}, c.kwargs)

    def test_args_and_kwargs(self):
        p = Curry(f, 5, 'baz', 6.7, answer=42, bar='foo')
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
        curry = Curry(self.call, *args, **kwargs)
        self.assertTrue(callable(curry))

    def test_call_called(self):
        curry = Curry(self.call)
        _ = curry()
        self.call.assert_called_once()
        self.call.assert_called_once_with()

    def test_call_called_with_instance_args(self):
        curry = Curry(self.call, 42, 'foo', 1.23)
        _ = curry()
        self.call.assert_called_once()
        self.call.assert_called_once_with(42, 'foo', 1.23)

    def test_call_called_with_instance_kwargs(self):
        curry = Curry(self.call, answer=42, bar='foo', susi=1.23)
        _ = curry()
        self.call.assert_called_once()
        self.call.assert_called_once_with(answer=42, bar='foo', susi=1.23)

    def test_call_called_with_instance_args_and_kwargs(self):
        args = 5, 'baz', 6.7
        kwargs = {'answer': 42, 'bar': 'foo', 'susi': 1.23}
        curry = Curry(self.call, *args, **kwargs)
        _ = curry()
        self.call.assert_called_once()
        self.call.assert_called_once_with(*args, **kwargs)

    def test_call_called_with_call_args(self):
        curry = Curry(self.call)
        _ = curry(42, 'foo', 1.23)
        self.call.assert_called_once()
        self.call.assert_called_once_with(42, 'foo', 1.23)

    def test_call_called_with_call_kwargs(self):
        curry = Curry(self.call)
        _ = curry(answer=42, bar='foo', susi=1.23)
        self.call.assert_called_once()
        self.call.assert_called_once_with(answer=42, bar='foo', susi=1.23)

    def test_call_called_with_call_args_and_kwargs(self):
        args = 5, 'baz', 6.7
        kwargs = {'answer': 42, 'bar': 'foo', 'susi': 1.23}
        curry = Curry(self.call)
        _ = curry(*args, **kwargs)
        self.call.assert_called_once()
        self.call.assert_called_once_with(*args, **kwargs)

    def test_call_arg_ordering(self):
        curry = Curry(self.call, 4, 'foo')
        _ = curry(2, 'bar')
        self.call.assert_called_once()
        self.call.assert_called_once_with( 2, 'bar', 4, 'foo')

    def test_call_kwarg_adding(self):
        curry = Curry(self.call, foo='bar')
        _ = curry(answer=42)
        self.call.assert_called_once()
        self.call.assert_called_once_with(foo='bar', answer=42)

    def test_call_kwarg_overwriting(self):
        curry = Curry(self.call, foo='bar', answer=49)
        _ = curry(answer=42)
        self.call.assert_called_once()
        self.call.assert_called_once_with(foo='bar', answer=42)

    def test_return_value(self):
        curry = Curry(f, 1)
        actual = curry(2)
        self.assertIsInstance(actual, int)
        self.assertEqual(3, actual)


class TestMisc(unittest.TestCase):

    def test_pickle_works(self):
        curry = Curry(f, 'foo', answer=42)
        _ = pickle.dumps(curry)

    def test_pickle_raises_lambda(self):
        curry = Curry(lambda x: x, 'foo', answer=42)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(curry)

    def test_lambda_repr(self):
        curry = Curry(lambda x: x, 'foo', answer=42)
        self.assertEqual("Curry(lambda, 'foo', answer=42)", repr(curry))

    def test_function_repr(self):
        curry = Curry(f, 'foo', answer=42)
        self.assertEqual("Curry(f, 'foo', answer=42)", repr(curry))

    def test_class_repr(self):
        curry = Curry(Cls, 'foo', answer=42)
        self.assertEqual("Curry(Cls, 'foo', answer=42)", repr(curry))

    def test_obj_repr(self):
        curry = Curry(Call(), 'foo', answer=42)
        self.assertEqual("Curry(Call(...), 'foo', answer=42)", repr(curry))

    def test_classmethod_repr(self):
        curry = Curry(Cls.m, 'foo', answer=42)
        self.assertEqual("Curry(Cls.m, 'foo', answer=42)", repr(curry))

    def test_staticmethod_repr(self):
        curry = Curry(Cls().s, 'foo', answer=42)
        self.assertEqual("Curry(Cls.s, 'foo', answer=42)", repr(curry))

    def test_method_repr(self):
        curry = Curry(Cls().m, 'foo', answer=42)
        self.assertEqual("Curry(Cls.m, 'foo', answer=42)", repr(curry))

    def test_argrepr(self):
        curry = Curry(A(1), A(2), answer=42)
        self.assertEqual("Curry(A(1), A(2), answer=42)", repr(curry))

    def test_indentrepr(self):
        curry = Curry(Ind([1, 2, 3]), Ind([1, 2, 3]), answer=42)
        expected = "Curry(Ind()[3], Ind()[3], answer=42)"
        self.assertEqual(expected, repr(curry))

    def test_type_annotation(self):
        _ = Curry[int](f, 42)


if __name__ == '__main__':
    unittest.main()

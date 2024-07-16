import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Safe, SafeError
from swak.magic import ArgRepr, IndentRepr


def f(x, y):
    return x + y


def g(x, y):
    return x / y


class Cls:

    @classmethod
    def c(cls, x, y):
        return x + y

    def m(self, x, y):
        _ = self.__class__.__name__
        return x.missing_attribute

    @staticmethod
    def s(x, y):
        return x + y + 'foo'


class Call:

    def __call__(self, x, y):
        return x + y


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __call__(self, x, y):
        return x / y


class Ind(IndentRepr):

    def __init__(self, *xs):
        super().__init__(*xs)

    def __call__(self, x, y):
        return x / y


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Safe(f)

    def test_has_call(self):
        s = Safe(f)
        self.assertTrue(hasattr(s, 'call'))

    def test_call_correct(self):
        s = Safe(f)
        self.assertIs(s.call, f)

    def test_has_exceptions(self):
        s = Safe(f)
        self.assertTrue(hasattr(s, 'exceptions'))

    def test_exceptions_correct(self):
        s = Safe(f)
        self.assertTupleEqual((Exception, ), s.exceptions)

    def test_decorator(self):

        @Safe
        def s(x):
            return x

        self.assertIsInstance(s, Safe)
        self.assertTrue(hasattr(s, 'call'))
        self.assertTrue(callable(s.call))
        self.assertTrue(hasattr(s, 'exceptions'))
        self.assertTupleEqual((Exception,), s.exceptions)


class TestDefaultUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Safe(f)))

    def test_called_once(self):
        mock = Mock()
        _ = Safe(mock)(1, 2)
        mock.assert_called_once()

    def test_called_correctly(self):
        mock = Mock()
        _ = Safe(mock)(1, 2)
        mock.assert_called_once_with(1, 2)

    def test_return_type_correct(self):
        actual = Safe(f)(1, 2)
        self.assertIsInstance(actual, int)

    def test_return_value_correct(self):
        actual = Safe(f)(1, 2)
        self.assertEqual(3, actual)

    def test_caught_correct_exception(self):
        actual = Safe(g)(1, 0)
        self.assertIsInstance(actual, SafeError)


class TestExceptionsAttributes(unittest.TestCase):

    def test_empty_exceptions(self):
        s = Safe(f, [])
        self.assertTrue(hasattr(s, 'call'))
        self.assertIs(s.call, f)
        self.assertTrue(hasattr(s, 'exceptions'))
        self.assertTupleEqual((Exception, ), s.exceptions)

    def test_list_two_exceptions(self):
        s = Safe(f, [ZeroDivisionError, StopIteration])
        self.assertTrue(hasattr(s, 'call'))
        self.assertIs(s.call, f)
        self.assertTrue(hasattr(s, 'exceptions'))
        self.assertTupleEqual((ZeroDivisionError, StopIteration), s.exceptions)

    def test_list_one_exception_one_exception(self):
        s = Safe(f, [ZeroDivisionError], StopIteration)
        self.assertTrue(hasattr(s, 'call'))
        self.assertIs(s.call, f)
        self.assertTrue(hasattr(s, 'exceptions'))
        self.assertTupleEqual((ZeroDivisionError, StopIteration), s.exceptions)

    def test_two_exceptions(self):
        s = Safe(f, ZeroDivisionError, StopIteration)
        self.assertTrue(hasattr(s, 'call'))
        self.assertIs(s.call, f)
        self.assertTrue(hasattr(s, 'exceptions'))
        self.assertTupleEqual((ZeroDivisionError, StopIteration), s.exceptions)


class TestExceptionsUsage(unittest.TestCase):

    def test_callable(self):
        self.assertTrue(callable(Safe(f, ZeroDivisionError, StopIteration)))

    def test_called_once(self):
        mock = Mock()
        _ = Safe(mock, ZeroDivisionError, StopIteration)(1, 2)
        mock.assert_called_once()

    def test_called_correctly(self):
        mock = Mock()
        _ = Safe(mock, ZeroDivisionError, StopIteration)(1, 2)
        mock.assert_called_once_with(1, 2)

    def test_return_type_correct(self):
        actual = Safe(f, ZeroDivisionError, StopIteration)(1, 2)
        self.assertIsInstance(actual, int)

    def test_return_value_correct(self):
        actual = Safe(f, ZeroDivisionError, StopIteration)(1, 2)
        self.assertEqual(3, actual)

    def test_caught_correct_exception(self):
        actual = Safe(g, ZeroDivisionError, StopIteration)(1, 0)
        self.assertIsInstance(actual, SafeError)

    def test_lets_wrong_exception_through(self):
        with self.assertRaises(AttributeError):
            _ = Safe(Cls().m, ZeroDivisionError, StopIteration)(1, 2)
        with self.assertRaises(TypeError):
            _ = Safe(Cls().s, ZeroDivisionError, StopIteration)(1, 2)


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        s = Safe(f)
        _ = pickle.dumps(s)

    def test_default_pickle_raises_lambda(self):
        s = Safe(lambda x, y: x + y)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(s)

    def test_acc_pickle_works(self):
        s = Safe(f, ZeroDivisionError, StopIteration)
        _ = pickle.dumps(s)

    def test_acc_pickle_raises_lambda(self):
        s = Safe(lambda x, y: x + y, ZeroDivisionError, StopIteration)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(s)

    def test_default_lambda_repr(self):
        s = Safe(lambda x, y: x + y)
        self.assertEqual('Safe(lambda)', repr(s))

    def test_default_function_repr(self):
        s = Safe(f)
        self.assertEqual('Safe(f)', repr(s))

    def test_default_class_repr(self):
        s = Safe(Cls)
        self.assertEqual('Safe(Cls)', repr(s))

    def test_default_obj_repr(self):
        s = Safe(Call())
        self.assertEqual('Safe(Call(...))', repr(s))

    def test_default_classmethod_repr(self):
        s = Safe(Cls.c)
        self.assertEqual('Safe(Cls.c)', repr(s))

    def test_default_staticmethod_repr(self):
        s = Safe(Cls().s)
        self.assertEqual('Safe(Cls.s)', repr(s))

    def test_default_method_repr(self):
        s = Safe(Cls().m)
        self.assertEqual('Safe(Cls.m)', repr(s))

    def test_default_argrepr(self):
        s = Safe(A(1))
        self.assertEqual("Safe(A(1))", repr(s))

    def test_default_indentrepr(self):
        s = Safe(Ind(1, 2, 3))
        self.assertEqual("Safe(Ind[3])", repr(s))

    def test_acc_repr(self):
        s = Safe(f, ZeroDivisionError)
        self.assertEqual('Safe(f, ZeroDivisionError)', repr(s))

    def test_acc_argrepr(self):
        s = Safe(A(1), ZeroDivisionError)
        self.assertEqual("Safe(A(1), ZeroDivisionError)", repr(s))

    def test_acc_indentrepr(self):
        s = Safe(Ind(1, 2, 3), ZeroDivisionError)
        self.assertEqual("Safe(Ind[3], ZeroDivisionError)", repr(s))

    def test_type_annotation(self):
        _ = Safe[[int], int](lambda x: x)

    def test_type_annotation_acc(self):
        _ = Safe[[int, float], int](f, ZeroDivisionError)


if __name__ == '__main__':
    unittest.main()

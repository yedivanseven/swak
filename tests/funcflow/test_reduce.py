import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Reduce
from swak.funcflow.exceptions import ReduceError
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
        return x + y

    @staticmethod
    def s(x, y):
        return x + y


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
        _ = Reduce(f)

    def test_has_call(self):
        r = Reduce(f)
        self.assertTrue(hasattr(r, 'call'))

    def test_call_correct(self):
        r = Reduce(f)
        self.assertIs(r.call, f)

    def test_has_acc(self):
        r = Reduce(f)
        self.assertTrue(hasattr(r, 'acc'))

    def test_acc_correct(self):
        r = Reduce(f)
        self.assertIsNone(r.acc)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.r = Reduce(f)

    def test_callable(self):
        self.assertTrue(callable(self.r))

    def test_no_call_single(self):
        mock = Mock()
        _ = Reduce(mock)([1])
        mock.assert_not_called()

    def test_called_once(self):
        mock = Mock()
        _ = Reduce(mock)([1, 2])
        mock.assert_called_once()

    def test_called_once_correctly(self):
        mock = Mock()
        _ = Reduce(mock)([1, 2])
        mock.assert_called_once_with(1, 2)

    def test_called_twice(self):
        mock = Mock()
        _ = Reduce(mock)([1, 2, 3])
        mock.assert_called()
        self.assertEqual(2, mock.call_count)

    def test_called_twice_correctly(self):
        mock = Mock(return_value=3)
        _ = Reduce(mock)([1, 2, 3])
        (a, _), (b, _) = mock.call_args_list
        self.assertTupleEqual((1, 2), a)
        self.assertTupleEqual((3, 3), b)

    def test_empty_raises(self):
        with self.assertRaises(StopIteration):
            _ = self.r([])

    def test_single_returns_first_element(self):
        actual = self.r([1])
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_return_correct(self):
        actual = self.r([1, 2, 3])
        self.assertIsInstance(actual, int)
        self.assertEqual(6, actual)

    def test_wrong_iterator_raises(self):
        with self.assertRaises(TypeError):
            _ = self.r(1)

    def test_wrong_call_raises(self):
        expected = ('\nZeroDivisionError calling\n'
                    'g\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(g)
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'A(1)\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(A(1))
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'Ind:\n'
                    '[ 0] 1\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(Ind(1))
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))


class TestAccAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Reduce(f, 1)

    def test_has_call(self):
        r = Reduce(f, 1)
        self.assertTrue(hasattr(r, 'call'))

    def test_call_correct(self):
        r = Reduce(f, 1)
        self.assertIs(r.call, f)

    def test_has_acc(self):
        r = Reduce(f, 1)
        self.assertTrue(hasattr(r, 'acc'))

    def test_acc_correct(self):
        cls = Cls()
        r = Reduce(f, cls)
        self.assertIs(r.acc, cls)


class TestAccUsage(unittest.TestCase):

    def setUp(self):
        self.r = Reduce(f, 1)

    def test_callable(self):
        self.assertTrue(callable(self.r))

    def test_no_call_empty(self):
        mock = Mock()
        _ = Reduce(mock, 1)([])
        mock.assert_not_called()

    def test_called_once(self):
        mock = Mock()
        _ = Reduce(mock, 1)([2])
        mock.assert_called_once()

    def test_called_once_correctly(self):
        mock = Mock()
        _ = Reduce(mock, 1)([2])
        mock.assert_called_once_with(1, 2)

    def test_called_twice(self):
        mock = Mock()
        _ = Reduce(mock, 1)([2, 3])
        mock.assert_called()
        self.assertEqual(2, mock.call_count)

    def test_called_twice_correctly(self):
        mock = Mock(return_value=3)
        _ = Reduce(mock, 1)([2, 3])
        (a, _), (b, _) = mock.call_args_list
        self.assertTupleEqual((1, 2), a)
        self.assertTupleEqual((3, 3), b)

    def test_empty_returns_acc(self):
        actual = self.r([])
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_return_correct(self):
        actual = self.r([2, 3])
        self.assertIsInstance(actual, int)
        self.assertEqual(6, actual)

    def test_wrong_iterator_raises(self):
        with self.assertRaises(TypeError):
            _ = self.r(1)

    def test_wrong_call_raises(self):
        expected = ('\nZeroDivisionError calling\n'
                    'g\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(g)
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'A(1)\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(A(1))
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'Ind:\n'
                    '[ 0] 1\n'
                    'on element #2:\n'
                    '0\n'
                    'float division by zero')
        r = Reduce(Ind(1))
        with self.assertRaises(ReduceError) as error:
            _ = r([1, 2, 0])
        self.assertEqual(expected, str(error.exception))


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        r = Reduce(f)
        _ = pickle.dumps(r)

    def test_default_pickle_raises_lambda(self):
        r = Reduce(lambda x, y: x + y)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(r)

    def test_acc_pickle_works(self):
        r = Reduce(f, 1)
        _ = pickle.dumps(r)

    def test_acc_pickle_raises_lambda(self):
        r = Reduce(lambda x, y: x + y, 1)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(r)

    def test_default_lambda_repr(self):
        r = Reduce(lambda x, y: x + y)
        self.assertEqual('Reduce(lambda, None)', repr(r))

    def test_default_function_repr(self):
        r = Reduce(f)
        self.assertEqual('Reduce(f, None)', repr(r))

    def test_default_class_repr(self):
        r = Reduce(Cls)
        self.assertEqual('Reduce(Cls, None)', repr(r))

    def test_default_obj_repr(self):
        r = Reduce(Call())
        self.assertEqual('Reduce(Call(...), None)', repr(r))

    def test_default_classmethod_repr(self):
        r = Reduce(Cls.c)
        self.assertEqual('Reduce(Cls.c, None)', repr(r))

    def test_default_staticmethod_repr(self):
        r = Reduce(Cls().s)
        self.assertEqual('Reduce(Cls.s, None)', repr(r))

    def test_default_method_repr(self):
        r = Reduce(Cls().m)
        self.assertEqual('Reduce(Cls.m, None)', repr(r))

    def test_default_argrepr(self):
        r = Reduce(A(1))
        self.assertEqual("Reduce(A(1), None)", repr(r))

    def test_default_indentrepr(self):
        r = Reduce(Ind(1, 2, 3))
        self.assertEqual("Reduce(Ind[3], None)", repr(r))

    def test_acc_repr(self):
        r = Reduce(f, 1)
        self.assertEqual('Reduce(f, 1)', repr(r))

    def test_acc_argrepr(self):
        r = Reduce(A(1), A(2))
        self.assertEqual("Reduce(A(1), A(2))", repr(r))

    def test_acc_indentrepr(self):
        r = Reduce(Ind(1, 2, 3), Ind(1, 2))
        self.assertEqual("Reduce(Ind[3], Ind[2])", repr(r))

    def test_type_annotation(self):
        _ = Reduce[int, str](lambda x: x)

    def test_type_annotation_acc(self):
        _ = Reduce[int, str](f, tuple)


if __name__ == '__main__':
    unittest.main()

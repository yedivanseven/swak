import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Filter
from swak.funcflow.exceptions import FilterError
from swak.misc import ArgRepr, IndentRepr


def g(x: int) -> bool:
    return x > 3


class Cls:

    @classmethod
    def c(cls, x: int) -> bool:
        return x > 3

    def m(self, x: int) -> bool:
        _ = self.__class__.__name__
        return x > 3

    @staticmethod
    def s(x: int) -> bool:
        return x > 3


class Call:

    def __call__(self, x: int) -> bool:
        return x > 3


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __bool__(self) -> bool:
        raise TypeError('Test!')

    def __call__(self, x: int) -> bool:
        return 1 / x


class Ind(IndentRepr):

    def __bool__(self) -> bool:
        raise TypeError('Test!')

    def __call__(self, x: int) -> bool:
        return 1 / x


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter()

    def test_has_criterion(self):
        f = Filter()
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_is_none(self):
        f = Filter()
        self.assertIsNone(f.criterion)

    def test_has_wrapper(self):
        f = Filter()
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_is_none(self):
        f = Filter()
        self.assertIsNone(f.wrapper)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.f = Filter()

    def test_callable(self):
        self.assertTrue(callable(self.f))

    def test_empty_list(self):
        actual = self.f([])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.f(())
        self.assertTupleEqual((), actual)

    def test_empty_set(self):
        actual = self.f(set())
        self.assertSetEqual(set(), actual)

    def test_list(self):
        actual = self.f([1, 2, 0, 2, 1])
        self.assertListEqual([1, 2, 2, 1], actual)

    def test_tuple(self):
        actual = self.f((1, 2, 0, 2, 1))
        self.assertTupleEqual((1, 2, 2, 1), actual)

    def test_set(self):
        actual = self.f({1, 2, 0, 2, 1})
        self.assertSetEqual({1, 2}, actual)

    def test_empty_result(self):
        actual = self.f([0, 0, 0])
        self.assertListEqual([], actual)

    def test_bool_criterion_raises_argrepr(self):
        expected = ('\nTypeError calling criterion\n'
                    'bool\n'
                    'on element #2:\n'
                    'A(1)\n'
                    'Test!')
        f = Filter()
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, A(1), 2, 1])
        self.assertEqual(expected, str(error.exception))

    def test_bool_criterion_raises_indentrepr(self):
        expected = ('\nTypeError calling criterion\n'
                    'bool\n'
                    'on element #2:\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    'Test!')
        f = Filter()
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, Ind([1]), 2, 1])
        self.assertEqual(expected, str(error.exception))

    def test_wrong_iterable_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "list_iterator\n"
                    "on filter results:\n"
                    "cannot create 'list_iterator' instances")
        with self.assertRaises(FilterError) as error:
            _ = self.f(iter([1, 2, 0, 2, 1]))
        self.assertEqual(expected, str(error.exception))


class TestCriterionAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter(g)

    def test_has_criterion(self):
        f = Filter(g)
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_correct(self):
        f = Filter(g)
        self.assertIs(f.criterion, g)

    def test_has_wrapper(self):
        f = Filter(g)
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_is_none(self):
        f = Filter(g)
        self.assertIsNone(f.wrapper)


class TestCriterionUsage(unittest.TestCase):

    def setUp(self):
        self.f = Filter(g)

    def test_callable(self):
        self.assertTrue(callable(self.f))

    def test_empty_list(self):
        actual = self.f([])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.f(())
        self.assertTupleEqual((), actual)

    def test_empty_set(self):
        actual = self.f(set())
        self.assertSetEqual(set(), actual)

    def test_criterion_not_called(self):
        mock = Mock()
        f = Filter(mock)
        _ = f([])
        mock.assert_not_called()

    def test_criterion_called(self):
        mock = Mock()
        f = Filter(mock)
        _ = f([1])
        mock.assert_called_once()

    def test_criterion_called_correctly(self):
        mock = Mock()
        f = Filter(mock)
        _ = f([1])
        mock.assert_called_once_with(1)

    def test_list(self):
        actual = self.f([1, 2, 3, 4, 5])
        self.assertListEqual([4, 5], actual)

    def test_tuple(self):
        actual = self.f((1, 2, 3, 4, 5))
        self.assertTupleEqual((4, 5), actual)

    def test_set(self):
        actual = self.f({1, 2, 3, 4, 5})
        self.assertSetEqual({4, 5}, actual)

    def test_wrong_iterable_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "list_iterator\n"
                    "on filter results:\n"
                    "cannot create 'list_iterator' instances")
        with self.assertRaises(FilterError) as error:
            _ = self.f(iter([1, 2, 3, 4, 5]))
        self.assertEqual(expected, str(error.exception))

    def test_wrong_criterion_raises(self):
        expected = ('\nZeroDivisionError calling criterion\n'
                    'lambda\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        f = Filter(lambda x: 1 / x)
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 0, 2, 1])
        self.assertEqual(expected, str(error.exception))

    def test_criterion_error_msg_argrepr(self):
        expected = ('\nZeroDivisionError calling criterion\n'
                    'A(1)\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        f = Filter(A(1))
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 0, 2, 1])
        self.assertEqual(expected, str(error.exception))

    def test_criterion_error_msg_indentrepr(self):
        expected = ('\nZeroDivisionError calling criterion\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    '[ 1] 2\n'
                    '[ 2] 3\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        f = Filter(Ind([1, 2, 3]))
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 0, 2, 1])
        self.assertEqual(expected, str(error.exception))


class TestWrapperAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Filter(g, tuple)

    def test_has_criterion(self):
        f = Filter(g, tuple)
        self.assertTrue(hasattr(f, 'criterion'))

    def test_criterion_correct(self):
        f = Filter(g, tuple)
        self.assertIs(f.criterion, g)

    def test_has_wrapper(self):
        f = Filter(g, tuple)
        self.assertTrue(hasattr(f, 'wrapper'))

    def test_wrapper_correct(self):
        f = Filter(g, tuple)
        self.assertIs(tuple, f.wrapper)


class TestWrapperUsage(unittest.TestCase):

    def test_callable(self):
        f = Filter(g, tuple)
        self.assertTrue(callable(f))

    def test_wrapper_called_once_empty(self):
        mock = Mock()
        f = Filter(g, mock)
        _ = f([])
        mock.assert_called_once()

    def test_wrapper_called_once(self):
        mock = Mock()
        f = Filter(g, mock)
        _ = f([1, 2, 3, 4, 5])
        mock.assert_called_once()

    def test_wrapper_called_once_correctly(self):
        mock = Mock()
        f = Filter(g, mock)
        _ = f([1, 2, 3, 4, 5])
        mock.assert_called_once_with([4, 5])

    def test_empty_list(self):
        f = Filter(g, tuple)
        actual = f([])
        self.assertTupleEqual((), actual)

    def test_empty_tuple(self):
        f = Filter(g, tuple)
        actual = f(())
        self.assertTupleEqual((), actual)

    def test_empty_set(self):
        f = Filter(g, tuple)
        actual = f(set())
        self.assertTupleEqual((), actual)

    def test_list(self):
        f = Filter(g, tuple)
        actual = f([1, 2, 3, 4, 5])
        self.assertTupleEqual((4, 5), actual)

    def test_tuple(self):
        f = Filter(g, tuple)
        actual = f((1, 2, 3, 4, 5))
        self.assertTupleEqual((4, 5), actual)

    def test_set(self):
        f = Filter(g, tuple)
        actual = f({1, 2, 3, 4, 5})
        self.assertTupleEqual((4, 5), actual)

    def test_wrong_wrapper_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "int\n"
                    "on filter results:\n"
                    "int() argument must be a string, a bytes-like"
                    " object or a real number, not 'list'")
        f = Filter(g, int)
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_argrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "A(1)\n"
                    "on filter results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        f = Filter(g, A(1))
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_indentrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "on filter results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        f = Filter(g, Ind([1]))
        with self.assertRaises(FilterError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))


class TestMisc(unittest.TestCase):

    def test_type_annotation_wrapper(self):
        _ = Filter[int, list](g)

    def test_type_annotation_wrapped_elements(self):
        _ = Filter[int, list[int]](g)

    def test_default_pickle_works(self):
        f = Filter()
        _ = pickle.loads(pickle.dumps(f))

    def test_default_repr(self):
        f = Filter()
        self.assertEqual('Filter(None, None)', repr(f))

    def test_criterion_pickle_works(self):
        f = Filter(g)
        _ = pickle.loads(pickle.dumps(f))

    def test_lambda_criterion_pickle_raises(self):
        f = Filter(lambda x: x > 3)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(f))

    def test_wrapper_pickle_works(self):
        f = Filter(g, tuple)
        _ = pickle.loads(pickle.dumps(f))

    def test_lambda_wrapper_pickle_raises(self):
        f = Filter(g, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(f))

    def test_criterion_lambda_repr(self):
        f = Filter(lambda x: x > 3)
        self.assertEqual('Filter(lambda, None)', repr(f))

    def test_criterion_function_repr(self):
        f = Filter(g)
        self.assertEqual('Filter(g, None)', repr(f))

    def test_criterion_class_repr(self):
        f = Filter(Cls)
        self.assertEqual('Filter(Cls, None)', repr(f))

    def test_criterion_obj_repr(self):
        f = Filter(Call())
        self.assertEqual('Filter(Call(...), None)', repr(f))

    def test_criterion_classmethod_repr(self):
        f = Filter(Cls.c)
        self.assertEqual('Filter(Cls.c, None)', repr(f))

    def test_criterion_staticmethod_repr(self):
        f = Filter(Cls().s)
        self.assertEqual('Filter(Cls.s, None)', repr(f))

    def test_criterion_method_repr(self):
        f = Filter(Cls().m)
        self.assertEqual('Filter(Cls.m, None)', repr(f))

    def test_criterion_argrepr_repr(self):
        f = Filter(A(1))
        self.assertEqual('Filter(A(1), None)', repr(f))

    def test_criterion_indentrepr(self):
        f = Filter(Ind([1, 2, 3]))
        self.assertEqual('Filter(Ind()[3], None)', repr(f))

    def test_wrapper_repr(self):
        f = Filter(wrapper=tuple)
        self.assertEqual('Filter(None, tuple)', repr(f))

    def test_wrapper_argrepr_repr(self):
        f = Filter(wrapper=A(1))
        self.assertEqual('Filter(None, A(1))', repr(f))

    def test_wrapper_indentrepr(self):
        f = Filter(wrapper=Ind([1, 2, 3]))
        self.assertEqual('Filter(None, Ind()[3])', repr(f))

    def test_criterion_wrapper_repr(self):
        f = Filter(g, tuple)
        self.assertEqual('Filter(g, tuple)', repr(f))


if __name__ == '__main__':
    unittest.main()

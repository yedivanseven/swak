import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Split
from swak.funcflow.exceptions import SplitError
from swak.magic import ArgRepr, IndentRepr


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

    def __init__(self, *xs):
        super().__init__(*xs)

    def __bool__(self) -> bool:
        raise TypeError('Test!')

    def __call__(self, x: int) -> bool:
        return 1 / x


class Ind(IndentRepr):

    def __init__(self, *xs):
        super().__init__(*xs)

    def __bool__(self) -> bool:
        raise TypeError('Test!')

    def __call__(self, x: int) -> bool:
        return 1 / x


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Split()

    def test_has_criterion(self):
        s = Split()
        self.assertTrue(hasattr(s, 'criterion'))

    def test_criterion_is_none(self):
        s = Split()
        self.assertIsNone(s.criterion)

    def test_has_wrapper(self):
        s = Split()
        self.assertTrue(hasattr(s, 'wrapper'))

    def test_wrapper_is_none(self):
        s = Split()
        self.assertIsNone(s.wrapper)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.s = Split()

    def test_callable(self):
        self.assertTrue(callable(self.s))

    def test_empty_list(self):
        actual = self.s([])
        self.assertTupleEqual(([], []), actual)

    def test_empty_tuple(self):
        actual = self.s(())
        self.assertTupleEqual(((), ()), actual)

    def test_empty_set(self):
        actual = self.s(set())
        self.assertTupleEqual((set(), set()), actual)

    def test_list(self):
        actual = self.s([1, 0, 0, 2])
        self.assertTupleEqual(([1, 2], [0, 0]), actual)

    def test_tuple(self):
        actual = self.s((1, 0, 0, 2))
        self.assertTupleEqual(((1, 2), (0, 0)), actual)

    def test_set(self):
        actual = self.s({1, 0, 0, 2})
        self.assertTupleEqual(({1, 2}, {0}), actual)

    def test_first_result_empty(self):
        actual = self.s([0, 0])
        self.assertTupleEqual(([], [0, 0]), actual)

    def test_second_result_empty(self):
        actual = self.s([1, 2])
        self.assertTupleEqual(([1, 2], []), actual)

    def test_bool_criterion_raises_argrepr(self):
        expected = ('\nTypeError calling\n'
                    'bool\n'
                    'on element #2:\n'
                    'A(1)\n'
                    'Test!')
        s = Split()
        with self.assertRaises(SplitError) as error:
            _ = s([1, 0, A(1), 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_bool_criterion_raises_indentrepr(self):
        expected = ('\nTypeError calling\n'
                    'bool\n'
                    'on element #2:\n'
                    'Ind:\n'
                    '[ 0] 1\n'
                    'Test!')
        s = Split()
        with self.assertRaises(SplitError) as error:
            _ = s([1, 0, Ind(1), 2, 0])
        self.assertEqual(expected, str(error.exception))

    def test_wrong_iterable_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "list_iterator\n"
                    "on split results:\n"
                    "cannot create 'list_iterator' instances")
        with self.assertRaises(SplitError) as error:
            _ = self.s(iter([1, 0, 0, 2]))
        self.assertEqual(expected, str(error.exception))


class TestCriterionAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Split(g)

    def test_has_criterion(self):
        s = Split(g)
        self.assertTrue(hasattr(s, 'criterion'))

    def test_criterion_correct(self):
        s = Split(g)
        self.assertIs(s.criterion, g)

    def test_has_wrapper(self):
        s = Split(g)
        self.assertTrue(hasattr(s, 'wrapper'))

    def test_wrapper_is_none(self):
        s = Split(g)
        self.assertIsNone(s.wrapper)


class TestCriterionUsage(unittest.TestCase):

    def setUp(self):
        self.s = Split(g)

    def test_callable(self):
        self.assertTrue(callable(self.s))

    def test_empty_list(self):
        actual = self.s([])
        self.assertTupleEqual(([], []), actual)

    def test_empty_tuple(self):
        actual = self.s(())
        self.assertTupleEqual(((), ()), actual)

    def test_empty_set(self):
        actual = self.s(set())
        self.assertTupleEqual((set(), set()), actual)

    def test_criterion_not_called(self):
        mock = Mock()
        s = Split(mock)
        _ = s([])
        mock.assert_not_called()

    def test_criterion_called(self):
        mock = Mock()
        s = Split(mock)
        _ = s([1])
        mock.assert_called_once()

    def test_criterion_called_correctly(self):
        mock = Mock()
        s = Split(mock)
        _ = s([1])
        mock.assert_called_once_with(1)

    def test_list(self):
        actual = self.s([1, 2, 3, 4, 5])
        self.assertTupleEqual(([4, 5], [1, 2, 3]), actual)

    def test_tuple(self):
        actual = self.s((1, 2, 3, 4, 5))
        self.assertTupleEqual(((4, 5), (1, 2, 3)), actual)

    def test_set(self):
        actual = self.s({1, 2, 3, 4, 5})
        self.assertTupleEqual(({4, 5}, {1, 2, 3}), actual)

    def test_wrong_iterable_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "list_iterator\n"
                    "on split results:\n"
                    "cannot create 'list_iterator' instances")
        with self.assertRaises(SplitError) as error:
            _ = self.s(iter([1, 2, 3, 4, 5]))
        self.assertEqual(expected, str(error.exception))

    def test_wrong_criterion_raises(self):
        expected = ('\nZeroDivisionError calling\n'
                    'lambda\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        s = Split(lambda x: 1 / x)
        with self.assertRaises(SplitError) as error:
            _ = s([1, 2, 0, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_criterion_error_msg_argrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'A(1, 2, 3)\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        s = Split(A(1, 2, 3))
        with self.assertRaises(SplitError) as error:
            _ = s([1, 2, 0, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_criterion_error_msg_indentrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'Ind:\n'
                    '[ 0] 1\n'
                    '[ 1] 2\n'
                    '[ 2] 3\n'
                    'on element #2:\n'
                    '0\n'
                    'division by zero')
        s = Split(Ind(1, 2, 3))
        with self.assertRaises(SplitError) as error:
            _ = s([1, 2, 0, 4, 5])
        self.assertEqual(expected, str(error.exception))


class TestWrapperAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Split(g, tuple)

    def test_has_criterion(self):
        s = Split(g, tuple)
        self.assertTrue(hasattr(s, 'criterion'))

    def test_criterion_correct(self):
        s = Split(g, tuple)
        self.assertIs(s.criterion, g)

    def test_has_wrapper(self):
        s = Split(g, tuple)
        self.assertTrue(hasattr(s, 'wrapper'))

    def test_wrapper_correct(self):
        s = Split(g, tuple)
        self.assertIs(tuple, s.wrapper)


class TestWrapperUsage(unittest.TestCase):

    def test_callable(self):
        s = Split(g, tuple)
        self.assertTrue(callable(s))

    def test_wrapper_called_twice_empty(self):
        mock = Mock()
        s = Split(g, mock)
        _ = s([])
        self.assertEqual(2, mock.call_count)

    def test_wrapper_called_twice(self):
        mock = Mock()
        s = Split(g, mock)
        _ = s([1, 2, 3, 4, 5])
        self.assertEqual(2, mock.call_count)

    def test_wrapper_called_twice_correctly(self):
        mock = Mock()
        s = Split(g, mock)
        _ = s([1, 2, 3, 4, 5])
        ((a,), _), ((b,), _) = mock.call_args_list
        self.assertListEqual([4, 5], a)
        self.assertListEqual([1, 2, 3], b)

    def test_empty_list(self):
        s = Split(g, tuple)
        actual = s([])
        self.assertTupleEqual(((), ()), actual)

    def test_empty_tuple(self):
        s = Split(g, tuple)
        actual = s(())
        self.assertTupleEqual(((), ()), actual)

    def test_empty_set(self):
        s = Split(g, tuple)
        actual = s(set())
        self.assertTupleEqual(((), ()), actual)

    def test_list(self):
        s = Split(g, tuple)
        actual = s([1, 2, 3, 4, 5])
        self.assertTupleEqual(((4, 5), (1, 2, 3)), actual)

    def test_tuple(self):
        s = Split(g, tuple)
        actual = s((1, 2, 3, 4, 5))
        self.assertTupleEqual(((4, 5), (1, 2, 3)), actual)

    def test_set(self):
        s = Split(g, tuple)
        actual = s({1, 2, 3, 4, 5})
        self.assertTupleEqual(((4, 5), (1, 2, 3)), actual)

    def test_wrong_wrapper_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "int\n"
                    "on split results:\n"
                    "int() argument must be a string, a bytes-like"
                    " object or a real number, not 'list'")
        s = Split(g, int)
        with self.assertRaises(SplitError) as error:
            _ = s([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_argrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "A(1)\n"
                    "on split results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        f = Split(g, A(1))
        with self.assertRaises(SplitError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_indentrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "Ind:\n"
                    "[ 0] 1\n"
                    "on split results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        f = Split(g, Ind(1))
        with self.assertRaises(SplitError) as error:
            _ = f([1, 2, 3, 4, 5])
        self.assertEqual(expected, str(error.exception))


class TestMisc(unittest.TestCase):

    def test_type_annotation_wrapper(self):
        _ = Split[int, list](g)

    def test_type_annotation_wrapped_elements(self):
        _ = Split[int, list[int]](g)

    def test_default_pickle_works(self):
        s = Split()
        _ = pickle.dumps(s)

    def test_default_repr(self):
        s = Split()
        self.assertEqual('Split(None, None)', repr(s))

    def test_criterion_pickle_works(self):
        s = Split(g)
        _ = pickle.dumps(s)

    def test_lambda_criterion_pickle_raises(self):
        s = Split(lambda x: x > 3)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(s)

    def test_wrapper_pickle_works(self):
        s = Split(g, tuple)
        _ = pickle.dumps(s)

    def test_lambda_wrapper_pickle_raises(self):
        s = Split(g, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(s)

    def test_criterion_lambda_repr(self):
        s = Split(lambda x: x > 3)
        self.assertEqual('Split(lambda, None)', repr(s))

    def test_criterion_function_repr(self):
        s = Split(g)
        self.assertEqual('Split(g, None)', repr(s))

    def test_criterion_class_repr(self):
        s = Split(Cls)
        self.assertEqual('Split(Cls, None)', repr(s))

    def test_criterion_obj_repr(self):
        s = Split(Call())
        self.assertEqual('Split(Call(...), None)', repr(s))

    def test_criterion_classmethod_repr(self):
        s = Split(Cls.c)
        self.assertEqual('Split(Cls.c, None)', repr(s))

    def test_criterion_staticmethod_repr(self):
        s = Split(Cls().s)
        self.assertEqual('Split(Cls.s, None)', repr(s))

    def test_criterion_method_repr(self):
        s = Split(Cls().m)
        self.assertEqual('Split(Cls.m, None)', repr(s))

    def test_criterion_argrepr_repr(self):
        s = Split(A(1))
        self.assertEqual('Split(A(1), None)', repr(s))

    def test_criterion_indentrepr(self):
        s = Split(Ind(1, 2, 3))
        self.assertEqual('Split(Ind[3], None)', repr(s))

    def test_wrapper_repr(self):
        s = Split(wrapper=tuple)
        self.assertEqual('Split(None, tuple)', repr(s))

    def test_wrapper_argrepr_repr(self):
        s = Split(wrapper=A(1))
        self.assertEqual('Split(None, A(1))', repr(s))

    def test_wrapper_indentrepr(self):
        s = Split(wrapper=Ind(1, 2, 3))
        self.assertEqual('Split(None, Ind[3])', repr(s))

    def test_criterion_wrapper_repr(self):
        s = Split(g, tuple)
        self.assertEqual('Split(g, tuple)', repr(s))


if __name__ == '__main__':
    unittest.main()

import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Map
from swak.funcflow.exceptions import MapError
from swak.misc import ArgRepr, IndentRepr


def plus_2(x: int) -> int:
    return x + 2


def plus(x: int, y: int) -> int:
    return x + y


def to_tuple(x: int) -> tuple:
    return x, x + 1


def to_empty_tuple(x: int) -> tuple:
    return ()


def to_single_tuple(x: int) -> tuple:
    return (x + 2,)


def to_list(x: int) -> list:
    return [x, x + 1]


def to_set(x: int) -> set:
    return {x, x + 1}


def to_dict(x: int) -> dict:
    return {x: x + 1}


def to_string(x: int) -> str:
    return 'ab'


def to_tuple_or_scalar(x: int) -> tuple | int:
    return (x, x + 1) if x % 2 == 0 else x


class Cls:

    @classmethod
    def c(cls, x: int) -> int:
        return x + 2

    def m(self, x: int) -> int:
        _ = self.__class__.__name__
        return x + 2

    @staticmethod
    def s(x: int) -> int:
        return x + 2


class Call:

    def __call__(self, x: int) -> int:
        return x + 2


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __call__(self, x: int) -> int:
        return 1 / x


class Ind(IndentRepr):

    def __call__(self, x: int) -> int:
        return 1 / x


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Map(plus_2)

    def test_has_transform(self):
        m = Map(plus_2)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = Map(plus_2)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = Map(plus_2)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = Map(plus_2)
        self.assertIsNone(m.wrapper)

    def test_has_flat(self):
        m = Map(plus_2)
        self.assertTrue(hasattr(m, 'flat'))

    def test_flat_is_false(self):
        m = Map(plus_2)
        self.assertIsInstance(m.flat, bool)
        self.assertFalse(m.flat)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = Map(plus_2)
        self.m2 = Map(plus)

    def test_callable(self):
        self.assertTrue(callable(self.m1))
        self.assertTrue(callable(self.m2))

    def test_empty_list(self):
        actual = self.m1([])
        self.assertListEqual([], actual)

    def test_empty_lists(self):
        actual = self.m1([], [])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.m1(())
        self.assertTupleEqual((), actual)

    def test_empty_tuples(self):
        actual = self.m1((), ())
        self.assertTupleEqual((), actual)

    def test_empty_set(self):
        actual = self.m1(set())
        self.assertSetEqual(set(), actual)

    def test_empty_sets(self):
        actual = self.m1(set(), set())
        self.assertSetEqual(set(), actual)

    def test_empty_mixed(self):
        actual = self.m1([], (), set())
        self.assertListEqual([], actual)

    def test_call_called(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1])
        mock.assert_called_once()

    def test_transform_called_correctly_one_iterable(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1])
        mock.assert_called_once_with(1)

    def test_transform_called_correctly_two_iterables(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1], (2,))
        mock.assert_called_once_with(1, 2)

    def test_transform_called_correctly_three_iterables(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1], (2,), {3})
        mock.assert_called_once_with(1, 2, 3)

    def test_list(self):
        actual = self.m1([1, 2, 3])
        self.assertListEqual([3, 4, 5], actual)

    def test_tuple(self):
        actual = self.m1((1, 2, 3))
        self.assertTupleEqual((3, 4, 5), actual)

    def test_lists(self):
        actual = self.m2([1, 2, 3], [1, 2, 3])
        self.assertListEqual([2, 4, 6], actual)

    def test_tuples(self):
        actual = self.m2((1, 2, 3), (1, 2, 3))
        self.assertTupleEqual((2, 4, 6), actual)

    def test_list_tuple(self):
        actual = self.m2([1, 2, 3], (1, 2, 3))
        self.assertListEqual([2, 4, 6], actual)

    def test_tuple_list(self):
        actual = self.m2((1, 2, 3), [1, 2, 3])
        self.assertTupleEqual((2, 4, 6), actual)

    def test_shortest_iterable(self):
        actual = self.m2([1, 2, 3], (1, 2, 3, 4, 5))
        self.assertListEqual([2, 4, 6], actual)

    def test_wrong_call_raises(self):
        expected = ('\nZeroDivisionError calling\n'
                    'lambda\n'
                    'on element #1:\n'
                    'division by zero')
        m = Map(lambda x: 1 / x)
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'A(1)\n'
                    'on element #1:\n'
                    'division by zero')
        m = Map(A(1))
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ('\nZeroDivisionError calling\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    'on element #1:\n'
                    'division by zero')
        m = Map(Ind([1]))
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_wrong_iterable_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "list_iterator\n"
                    "on map results:\n"
                    "cannot create 'list_iterator' instances")
        with self.assertRaises(MapError) as error:
            _ = self.m1(iter([1, 2, 3]))
        self.assertEqual(expected, str(error.exception))


class TestWrapperAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Map(plus_2, tuple)

    def test_has_transform(self):
        m = Map(plus_2, tuple)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = Map(plus_2, tuple)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = Map(plus_2, tuple)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_correct(self):
        m = Map(plus_2, tuple)
        self.assertIs(m.wrapper, tuple)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = Map(plus_2, list)
        self.m2 = Map(plus, list)

    def test_callable(self):
        self.assertTrue(callable(self.m1))
        self.assertTrue(callable(self.m2))

    def test_wrapper_called(self):
        mock = Mock()
        m = Map(plus_2, mock)
        _ = m([1, 2, 3])
        mock.assert_called_once()

    def test_wrapper_called_correctly(self):
        mock = Mock()
        m = Map(plus_2, mock)
        _ = m([1, 2, 3])
        mock.assert_called_once_with([3, 4, 5])

    def test_list(self):
        actual = self.m1([1, 2, 3])
        self.assertListEqual([3, 4, 5], actual)

    def test_tuple(self):
        actual = self.m1((1, 2, 3))
        self.assertListEqual([3, 4, 5], actual)

    def test_lists(self):
        actual = self.m2([1, 2, 3], [1, 2, 3])
        self.assertListEqual([2, 4, 6], actual)

    def test_tuples(self):
        actual = self.m2((1, 2, 3), (1, 2, 3))
        self.assertListEqual([2, 4, 6], actual)

    def test_list_tuple(self):
        actual = self.m2([1, 2, 3], (1, 2, 3))
        self.assertListEqual([2, 4, 6], actual)

    def test_tuple_list(self):
        actual = self.m2((1, 2, 3), [1, 2, 3])
        self.assertListEqual([2, 4, 6], actual)

    def test_shortest_iterable(self):
        actual = self.m2([1, 2, 3], (1, 2, 3, 4, 5))
        self.assertListEqual([2, 4, 6], actual)

    def test_wrong_wrapper_raises(self):
        expected = ("\nTypeError calling wrapper\n"
                    "int\n"
                    "on map results:\n"
                    "int() argument must be a string, a bytes-like"
                    " object or a real number, not 'list'")
        m = Map(plus_2, int)
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_argrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "A(1)\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = Map(plus_2, A(1))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_indentrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = Map(plus_2, Ind([1]))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))


class TestFlatAttributes(unittest.TestCase):

    def test_has_flat(self):
        m = Map(plus_2, None, True)
        self.assertTrue(hasattr(m, 'flat'))

    def test_flat_is_true(self):
        m = Map(plus_2, None, True)
        self.assertIsInstance(m.flat, bool)
        self.assertTrue(m.flat)

    def test_flat_with_wrapper_is_true(self):
        m = Map(plus_2, list, True)
        self.assertIsInstance(m.flat, bool)
        self.assertTrue(m.flat)


class TestFlatUsage(unittest.TestCase):

    def setUp(self):
        self.m = Map(to_tuple, None, True)

    def test_non_tuple_result_not_flattened(self):
        m = Map(plus_2, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([3, 4, 5], actual)

    def test_empty_tuple_result_disappears(self):
        m = Map(to_empty_tuple, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([], actual)

    def test_single_element_tuple_unpacked(self):
        m = Map(to_single_tuple, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([3, 4, 5], actual)

    def test_multi_element_tuple_unpacked(self):
        actual = self.m([1, 2, 3])
        self.assertListEqual([1, 2, 2, 3, 3, 4], actual)

    def test_output_longer_than_input(self):
        actual = self.m([1, 2, 3])
        self.assertGreater(len(actual), 3)

    def test_mixed_tuple_and_scalar(self):
        m = Map(to_tuple_or_scalar, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([1, 2, 3, 3], actual)

    def test_empty_input_flat_true(self):
        actual = self.m([])
        self.assertListEqual([], actual)

    def test_empty_tuple_input_flat_true(self):
        actual = self.m(())
        self.assertTupleEqual((), actual)

    def test_wrapper_respected_with_flat(self):
        m = Map(to_tuple, tuple, True)
        actual = m([1, 2, 3])
        self.assertTupleEqual((1, 2, 2, 3, 3, 4), actual)

    def test_string_result_not_flattened(self):
        m = Map(to_string, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual(['ab', 'ab', 'ab'], actual)

    def test_list_result_not_flattened(self):
        m = Map(to_list, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([[1, 2], [2, 3], [3, 4]], actual)

    def test_set_result_not_flattened(self):
        m = Map(to_set, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([{1, 2}, {2, 3}, {3, 4}], actual)

    def test_dict_result_not_flattened(self):
        m = Map(to_dict, None, True)
        actual = m([1, 2, 3])
        self.assertListEqual([{1: 2}, {2: 3}, {3: 4}], actual)


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        m = Map(plus_2)
        _ = pickle.loads(pickle.dumps(m))

    def test_default_pickle_raises_lambda(self):
        m = Map(lambda x: x + 2)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_wrapper_pickle_works(self):
        m = Map(plus_2, tuple)
        _ = pickle.loads(pickle.dumps(m))

    def test_wrapper_pickle_raises_with_lambda(self):
        m = Map(plus_2, lambda x: tuple(x))  #noqa
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_pickle_raises_with_lambda(self):
        m = Map(lambda x: x + 2, lambda x: tuple(x))  #noqa
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_default_lambda_repr(self):
        m = Map(lambda x: x > 3)
        self.assertEqual('Map(lambda, None, False)', repr(m))

    def test_default_function_repr(self):
        m = Map(plus_2)
        self.assertEqual('Map(plus_2, None, False)', repr(m))

    def test_default_class_repr(self):
        m = Map(Cls)
        self.assertEqual('Map(Cls, None, False)', repr(m))

    def test_default_obj_repr(self):
        m = Map(Call())
        self.assertEqual('Map(Call(...), None, False)', repr(m))

    def test_default_classmethod_repr(self):
        m = Map(Cls.c)
        self.assertEqual('Map(Cls.c, None, False)', repr(m))

    def test_default_staticmethod_repr(self):
        m = Map(Cls().s)
        self.assertEqual('Map(Cls.s, None, False)', repr(m))

    def test_default_method_repr(self):
        m = Map(Cls().m)
        self.assertEqual('Map(Cls.m, None, False)', repr(m))

    def test_default_argrepr(self):
        m = Map(A(1))
        self.assertEqual('Map(A(1), None, False)', repr(m))

    def test_default_indentrepr(self):
        m = Map(Ind([1, 2, 3]))
        self.assertEqual('Map(Ind()[3], None, False)', repr(m))

    def test_wrapper_repr(self):
        m = Map(plus_2, tuple)
        self.assertEqual('Map(plus_2, tuple, False)', repr(m))

    def test_wrapper_argrepr(self):
        m = Map(plus_2, A(1))
        self.assertEqual('Map(plus_2, A(1), False)', repr(m))

    def test_wrapper_indentrepr(self):
        m = Map(plus_2, Ind([1, 2, 3]))
        self.assertEqual('Map(plus_2, Ind()[3], False)', repr(m))

    def test_type_annotation_wrapper(self):
        _ = Map[[int, bool], float, list]

    def test_type_annotation_wrapped_elements(self):
        _ = Map[[int, bool], float, list[float]]


if __name__ == '__main__':
    unittest.main()

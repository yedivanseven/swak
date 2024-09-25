import unittest
import pickle
from unittest.mock import Mock, patch
from swak.funcflow.concurrent import ProcessMap
from swak.funcflow.exceptions import MapError
from swak.misc import ArgRepr, IndentRepr


def plus_2(x: int) -> int:
    return x + 2


def plus(x: int, y: int) -> int:
    return x + y


def f(x: int) -> float:
    return 1 / x


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

    def __call__(self, x: int) -> float:
        return 1 / x


class Ind(IndentRepr):

    def __call__(self, x: int) -> float:
        return 1 / x


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ProcessMap(plus_2)

    def test_has_transform(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = ProcessMap(plus_2)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = ProcessMap(plus_2)
        self.assertIsNone(m.wrapper)

    def test_has_max_workers(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'max_workers'))

    def test_max_workers_correct(self):
        m = ProcessMap(plus_2)
        self.assertIsInstance(m.max_workers, int)
        self.assertGreater(m.max_workers, 0)

    def test_has_max_tasks_per_child(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'max_tasks_per_child'))

    def test_max_tasks_per_child_is_none(self):
        m = ProcessMap(plus_2)
        self.assertIsNone(m.max_tasks_per_child)

    def test_has_initializer(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'initializer'))

    def test_initializer_is_none(self):
        m = ProcessMap(plus_2)
        self.assertIsNone(m.initializer)

    def test_has_initargs(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'initargs'))

    def test_initargs_correct(self):
        m = ProcessMap(plus_2)
        self.assertTupleEqual((), m.initargs)

    def test_has_timeout(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_is_none(self):
        m = ProcessMap(plus_2)
        self.assertIsNone(m.timeout)

    def test_has_chunksize(self):
        m = ProcessMap(plus_2)
        self.assertTrue(hasattr(m, 'chunksize'))

    def test_chunksize_correct(self):
        m = ProcessMap(plus_2)
        self.assertIsInstance(m.chunksize, int)
        self.assertEqual(1, m.chunksize)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = ProcessMap(plus_2)
        self.m2 = ProcessMap(plus)

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
        expected = ('Error calling\n'
                    'f\n'
                    'on one or more element(s) of the iterable(s)!\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        m = ProcessMap(f)
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ('Error calling\n'
                    'A(1)\n'
                    'on one or more element(s) of the iterable(s)!\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        m = ProcessMap(A(1))
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ('Error calling\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    'on one or more element(s) of the iterable(s)!\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        m = ProcessMap(Ind([1]))
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
        _ = ProcessMap(plus_2, tuple)

    def test_has_wrapper(self):
        m = ProcessMap(plus_2, tuple)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_correct(self):
        m = ProcessMap(plus_2, tuple)
        self.assertIs(m.wrapper, tuple)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = ProcessMap(plus_2, list)
        self.m2 = ProcessMap(plus, list)

    def test_callable(self):
        self.assertTrue(callable(self.m1))
        self.assertTrue(callable(self.m2))

    def test_wrapper_called(self):
        mock = Mock()
        m = ProcessMap(plus_2, mock)
        _ = m([1, 2, 3])
        mock.assert_called_once()

    def test_wrapper_called_correctly(self):
        mock = Mock()
        m = ProcessMap(plus_2, mock)
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
        m = ProcessMap(plus_2, int)
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_argrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "A(1)\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = ProcessMap(plus_2, A(1))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_indentrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = ProcessMap(plus_2, Ind([1]))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))


class TestProcessPoolAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)

    def test_has_transform(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertIsNone(m.wrapper)

    def test_has_max_workers(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'max_workers'))

    def test_max_workers_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertIsInstance(m.max_workers, int)
        self.assertEqual(8, m.max_workers)

    def test_has_max_tasks_per_child(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'max_tasks_per_child'))

    def test_max_tasks_per_child_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertIsInstance(m.max_tasks_per_child, int)
        self.assertEqual(3, m.max_tasks_per_child)

    def test_has_initializer(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'initializer'))

    def test_initializer_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertIs(m.initializer, plus)

    def test_has_initargs(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTrue(hasattr(m, 'initargs'))

    def test_initargs_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        self.assertTupleEqual((1, 2), m.initargs)


class TestProcessPoolUsage(unittest.TestCase):

    @patch('swak.funcflow.concurrent.processmap.ProcessPoolExecutor')
    def test_processpool_called(self, cls):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        _ = m([1, 2, 3])
        cls.assert_called_once()

    @patch('swak.funcflow.concurrent.processmap.ProcessPoolExecutor')
    def test_processpool_called_with_processpoolargs(self, cls):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        _ = m([1, 2, 3])
        cls.assert_called_once_with(
            8,
            None,
            plus,
            (1, 2),
            max_tasks_per_child=3
        )


class TestMapAttributes(unittest.TestCase):

    def test_has_timeout_arg(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3, 42)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_arg_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3, Cls)
        self.assertIs(m.timeout, Cls)

    def test_has_chunksize_arg(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3, 42, 5)
        self.assertTrue(hasattr(m, 'chunksize'))

    def test_timeout_chunksize_correct(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3, 42, Cls)
        self.assertIs(m.chunksize, Cls)

    def test_has_timeout_kwarg(self):
        m = ProcessMap(plus_2, timeout=42)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_kwarg_correct(self):
        m = ProcessMap(plus_2, timeout=Cls)
        self.assertIs(m.timeout, Cls)

    def test_has_chunksize_kwarg(self):
        m = ProcessMap(plus_2, chunksize=42)
        self.assertTrue(hasattr(m, 'chunksize'))

    def test_chunksize_kwarg_correct(self):
        m = ProcessMap(plus_2, chunksize=Cls)
        self.assertIs(m.chunksize, Cls)


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        m = ProcessMap(plus_2)
        _ = pickle.dumps(m)

    def test_default_pickle_raises_lambda(self):
        m = ProcessMap(lambda x: x + 2)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_wrapper_pickle_works(self):
        m = ProcessMap(plus_2, tuple)
        _ = pickle.dumps(m)

    def test_wrapper_pickle_raises_lambda(self):
        m = ProcessMap(plus_2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_pickle_raises_lambda(self):
        m = ProcessMap(lambda x: x + 2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_default_lambda_repr(self):
        m = ProcessMap(lambda x: x > 3)
        expected = "ProcessMap(lambda, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_function_repr(self):
        m = ProcessMap(plus_2)
        expected = "ProcessMap(plus_2, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_class_repr(self):
        m = ProcessMap(Cls)
        expected = "ProcessMap(Cls, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_obj_repr(self):
        m = ProcessMap(Call())
        expected = "ProcessMap(Call(...), None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_classmethod_repr(self):
        m = ProcessMap(Cls.c)
        expected = "ProcessMap(Cls.c, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_staticmethod_repr(self):
        m = ProcessMap(Cls().s)
        expected = "ProcessMap(Cls.s, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_method_repr(self):
        m = ProcessMap(Cls().m)
        expected = "ProcessMap(Cls.m, None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_default_argrepr(self):
        m = ProcessMap(A(1))
        excepted = "ProcessMap(A(1), None, 4, None, (), None, None, 1)"
        self.assertEqual(excepted, repr(m))

    def test_default_indentrepr(self):
        m = ProcessMap(Ind([1, 2, 3]))
        expected = "ProcessMap(Ind()[3], None, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_repr(self):
        m = ProcessMap(plus_2, tuple)
        expected = "ProcessMap(plus_2, tuple, 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_argrepr(self):
        m = ProcessMap(plus_2, A(1))
        expected = "ProcessMap(plus_2, A(1), 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_indentrepr(self):
        m = ProcessMap(plus_2, Ind([1, 2, 3]))
        expected = "ProcessMap(plus_2, Ind()[3], 4, None, (), None, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_processpool_repr(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3)
        expected = "ProcessMap(plus_2, None, 8, plus, (1, 2), 3, None, 1)"
        self.assertEqual(expected, repr(m))

    def test_map_arg_repr(self):
        m = ProcessMap(plus_2, None, 8, plus, (1, 2), 3, 42, 5)
        expect = "ProcessMap(plus_2, None, 8, plus, (1, 2), 3, 42, 5)"
        self.assertEqual(expect, repr(m))

    def test_map_kwarg_repr(self):
        m = ProcessMap(plus_2, timeout=42, chunksize=5)
        expected = "ProcessMap(plus_2, None, 4, None, (), None, 42, 5)"
        self.assertEqual(expected, repr(m))

    def test_type_annotation_wrapper(self):
        _ = ProcessMap[[int, bool], float, list]

    def test_type_annotation_wrapped_elements(self):
        _ = ProcessMap[[int, bool], float, list[float]]


if __name__ == '__main__':
    unittest.main()

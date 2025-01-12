import unittest
import pickle
from unittest.mock import Mock, patch
from swak.funcflow.concurrent import ThreadMap
from swak.funcflow.exceptions import MapError
from swak.misc import ArgRepr, IndentRepr


def plus_2(x: int) -> int:
    return x + 2


def plus(x: int, y: int) -> int:
    return x + y


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

    def __init__(self, *xs):
        super().__init__(*xs)

    def __call__(self, x: int) -> int:
        return 1 / x


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ThreadMap(plus_2)

    def test_has_transform(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = ThreadMap(plus_2)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = ThreadMap(plus_2)
        self.assertIsNone(m.wrapper)

    def test_has_max_workers(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'max_workers'))

    def test_max_workers_correct(self):
        m = ThreadMap(plus_2)
        self.assertIsInstance(m.max_workers, int)
        self.assertGreater(m.max_workers, 0)

    def test_has_thread_name_prefix(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'thread_name_prefix'))

    def test_thread_name_prefix_correct(self):
        m = ThreadMap(plus_2)
        self.assertIsInstance(m.thread_name_prefix, str)

    def test_has_initializer(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'initializer'))

    def test_initializer_is_none(self):
        m = ThreadMap(plus_2)
        self.assertIsNone(m.initializer)

    def test_has_initargs(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'initargs'))

    def test_initargs_correct(self):
        m = ThreadMap(plus_2)
        self.assertTupleEqual((), m.initargs)

    def test_has_timeout(self):
        m = ThreadMap(plus_2)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_is_none(self):
        m = ThreadMap(plus_2)
        self.assertIsNone(m.timeout)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = ThreadMap(plus_2)
        self.m2 = ThreadMap(plus)

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
        m = ThreadMap(mock)
        _ = m([1])
        mock.assert_called_once()

    def test_transform_called_correctly_one_iterable(self):
        mock = Mock()
        m = ThreadMap(mock)
        _ = m([1])
        mock.assert_called_once_with(1)

    def test_transform_called_correctly_two_iterables(self):
        mock = Mock()
        m = ThreadMap(mock)
        _ = m([1], (2,))
        mock.assert_called_once_with(1, 2)

    def test_transform_called_correctly_three_iterables(self):
        mock = Mock()
        m = ThreadMap(mock)
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
        expected = ('Error calling\n'
                    'lambda\n'
                    'on one or more element(s) of the iterable(s)!\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        m = ThreadMap(lambda x: 1 / x)
        with self.assertRaises(MapError) as error:
            _ = m([1, 0, 2])
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ('Error calling\n'
                    'A(1)\n'
                    'on one or more element(s) of the iterable(s)!\n'
                    'ZeroDivisionError:\n'
                    'division by zero')
        m = ThreadMap(A(1))
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
        m = ThreadMap(Ind([1]))
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
        _ = ThreadMap(plus_2, tuple)

    def test_has_wrapper(self):
        m = ThreadMap(plus_2, tuple)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_correct(self):
        m = ThreadMap(plus_2, tuple)
        self.assertIs(m.wrapper, tuple)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = ThreadMap(plus_2, list)
        self.m2 = ThreadMap(plus, list)

    def test_callable(self):
        self.assertTrue(callable(self.m1))
        self.assertTrue(callable(self.m2))

    def test_wrapper_called(self):
        mock = Mock()
        m = ThreadMap(plus_2, mock)
        _ = m([1, 2, 3])
        mock.assert_called_once()

    def test_wrapper_called_correctly(self):
        mock = Mock()
        m = ThreadMap(plus_2, mock)
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
        m = ThreadMap(plus_2, int)
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_argrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "A(1)\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = ThreadMap(plus_2, A(1))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))

    def test_wrapper_error_msg_indentrepr(self):
        expected = ("\nTypeError calling wrapper\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "on map results:\n"
                    "unsupported operand type(s) for /: 'int' and 'list'")
        m = ThreadMap(plus_2, Ind([1]))
        with self.assertRaises(MapError) as error:
            _ = m([1, 2, 3])
        self.assertEqual(expected, str(error.exception))


class TestThreadPoolAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))

    def test_has_transform(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertIsNone(m.wrapper)

    def test_has_max_workers(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'max_workers'))

    def test_max_workers_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertIsInstance(m.max_workers, int)
        self.assertEqual(8, m.max_workers)

    def test_has_thread_name_prefix(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'thread_name_prefix'))

    def test_thread_name_prefix_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertEqual('tm', m.thread_name_prefix)

    def test_has_initializer(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'initializer'))

    def test_initializer_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertIs(m.initializer, plus)

    def test_has_initargs(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTrue(hasattr(m, 'initargs'))

    def test_initargs_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        self.assertTupleEqual((1, 2), m.initargs)


class TestThreadPoolUsage(unittest.TestCase):

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor')
    def test_threadpool_called(self, cls):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        _ = m([1, 2, 3])
        cls.assert_called_once()

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor')
    def test_threadpool_called_with_threadpoolargs(self, cls):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        _ = m([1, 2, 3])
        cls.assert_called_once_with(8, 'tm', plus, (1, 2))


class TestMapAttributes(unittest.TestCase):

    def test_has_timeout_arg(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2), 42)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_arg_correct(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2), Cls)
        self.assertIs(m.timeout, Cls)

    def test_has_timeout_kwarg(self):
        m = ThreadMap(plus_2, timeout=42)
        self.assertTrue(hasattr(m, 'timeout'))

    def test_timeout_kwarg_correct(self):
        m = ThreadMap(plus_2, timeout=Cls)
        self.assertIs(m.timeout, Cls)


class TestMapUsage(unittest.TestCase):

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor.map')
    def test_map_called(self, method):
        m = ThreadMap(plus_2)
        _ = m([1, 2, 3])
        method.assert_called_once()

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor.map')
    def test_map_called_with_iterable_no_timeout(self, method):
        m = ThreadMap(plus_2)
        _ = m([1, 2, 3])
        method.assert_called_once_with(plus_2, [1, 2, 3], timeout=None)

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor.map')
    def test_map_called_with_iterables_no_timeout(self, method):
        m = ThreadMap(plus)
        _ = m([1, 2, 3], (1, 2, 3))
        method.assert_called_once_with(
            plus,
            [1, 2, 3],
            (1, 2, 3),
            timeout=None
        )

    @patch('swak.funcflow.concurrent.threadmap.ThreadPoolExecutor.map')
    def test_map_called_with_timeout(self, method):
        m = ThreadMap(plus_2, timeout=42)
        _ = m([1, 2, 3])
        method.assert_called_once_with(plus_2, [1, 2, 3], timeout=42)


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        m = ThreadMap(plus_2)
        _ = pickle.dumps(m)

    def test_default_pickle_raises_with_lambda(self):
        m = ThreadMap(lambda x: x + 2)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_wrapper_pickle_works(self):
        m = ThreadMap(plus_2, tuple)
        _ = pickle.loads(pickle.dumps(m))

    def test_wrapper_pickle_raises_lambda(self):
        m = ThreadMap(plus_2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_pickle_raises_with_lambda(self):
        m = ThreadMap(lambda x: x + 2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(m))

    def test_default_lambda_repr(self):
        m = ThreadMap(lambda x: x > 3)
        expected = "ThreadMap(lambda, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_function_repr(self):
        m = ThreadMap(plus_2)
        expected = "ThreadMap(plus_2, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_class_repr(self):
        m = ThreadMap(Cls)
        expected = "ThreadMap(Cls, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_obj_repr(self):
        m = ThreadMap(Call())
        expected = "ThreadMap(Call(...), None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_classmethod_repr(self):
        m = ThreadMap(Cls.c)
        expected = "ThreadMap(Cls.c, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_staticmethod_repr(self):
        m = ThreadMap(Cls().s)
        expected = "ThreadMap(Cls.s, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_method_repr(self):
        m = ThreadMap(Cls().m)
        expected = "ThreadMap(Cls.m, None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_default_argrepr(self):
        m = ThreadMap(A(1))
        excepted = "ThreadMap(A(1), None, 16, '', None, (), None)"
        self.assertEqual(excepted, repr(m))

    def test_default_indentrepr(self):
        m = ThreadMap(Ind([1, 2, 3]))
        expected = "ThreadMap(Ind()[3], None, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_repr(self):
        m = ThreadMap(plus_2, tuple)
        expected = "ThreadMap(plus_2, tuple, 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_argrepr(self):
        m = ThreadMap(plus_2, A(1))
        expected = "ThreadMap(plus_2, A(1), 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_wrapper_indentrepr(self):
        m = ThreadMap(plus_2, Ind([1, 2, 3]))
        expected = "ThreadMap(plus_2, Ind()[3], 16, '', None, (), None)"
        self.assertEqual(expected, repr(m))

    def test_threadpool_repr(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2))
        expected = "ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2), None)"
        self.assertEqual(expected, repr(m))

    def test_map_arg_repr(self):
        m = ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2), 42)
        expected = "ThreadMap(plus_2, None, 8, 'tm', plus, (1, 2), 42)"
        self.assertEqual(expected, repr(m))

    def test_map_kwarg_repr(self):
        m = ThreadMap(plus_2, timeout=42)
        expected = "ThreadMap(plus_2, None, 16, '', None, (), 42)"
        self.assertEqual(expected, repr(m))

    def test_type_annotation_wrapper(self):
        _ = ThreadMap[[int, bool], float, list]

    def test_type_annotation_wrapped_elements(self):
        _ = ThreadMap[[int, bool], float, list[float]]


if __name__ == '__main__':
    unittest.main()

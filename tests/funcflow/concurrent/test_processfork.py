import unittest
import pickle
from unittest.mock import patch
from swak.funcflow.concurrent import ProcessFork
from swak.funcflow.exceptions import ForkError
from swak.magic import ArgRepr, IndentRepr


def f():
    return ()


def g(*_):
    raise AttributeError('Test!')


def h(x, y):
    return x + y


def no_return_value(*_):
    return ()


def return_value(*_):
    return 1


def return_values(*_):
    return 1, 2


class Cls:

    @classmethod
    def c(cls):
        pass

    def m(self):
        pass

    @staticmethod
    def s():
        pass


class Call:

    def __call__(self):
        pass


class A(ArgRepr):

    def __init__(self, a):
        super().__init__(a)
        self.a = a

    def __call__(self, *xs):
        raise AttributeError('Test!')


class Ind(IndentRepr):

    def __call__(self, *ys):
        raise AttributeError('Test!')


class TestAttributes(unittest.TestCase):

    def default_attributes(self, fork):
        self.assertTrue(hasattr(fork, 'max_workers'))
        self.assertIsInstance(fork.max_workers, int)
        self.assertGreater(fork.max_workers, 0)
        self.assertTrue(hasattr(fork, 'initializer'))
        self.assertIsNone(fork.initializer)
        self.assertTrue(hasattr(fork, 'initargs'))
        self.assertTupleEqual((), fork.initargs)
        self.assertTrue(hasattr(fork, 'max_tasks_per_child'))
        self.assertIsNone(fork.max_tasks_per_child)
        self.assertTrue(hasattr(fork, 'timeout'))
        self.assertIsNone(fork.timeout)

    def test_empty(self):
        fork = ProcessFork()
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((), fork.calls)
        self.default_attributes(fork)

    def test_empty_list(self):
        fork = ProcessFork([])
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((), fork.calls)
        self.default_attributes(fork)

    def test_function(self):
        fork = ProcessFork(f)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, ), fork.calls)
        self.default_attributes(fork)

    def test_functions(self):
        fork = ProcessFork(f, g)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g), fork.calls)
        self.default_attributes(fork)

    def test_lambda(self):
        fork = ProcessFork(lambda x: x)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertEqual(1, len(fork.calls))
        self.default_attributes(fork)

    def test_lambdas(self):
        fork = ProcessFork(lambda x: x, lambda y: ())
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertEqual(2, len(fork.calls))
        self.default_attributes(fork)

    def test_class(self):
        fork = ProcessFork(Cls)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls,), fork.calls)
        self.default_attributes(fork)

    def test_classes(self):
        fork = ProcessFork(Cls, Call, A)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls, Call, A), fork.calls)
        self.default_attributes(fork)

    def test_object(self):
        call = Call()
        fork = ProcessFork(call)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((call,), fork.calls)
        self.default_attributes(fork)

    def test_method(self):
        cls = Cls()
        fork = ProcessFork(cls.m)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((cls.m,), fork.calls)
        self.default_attributes(fork)

    def test_methods(self):
        cls = Cls()
        fork = ProcessFork(Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls.c, cls.m, cls.s), fork.calls)
        self.default_attributes(fork)

    def test_mix(self):
        cls = Cls()
        call = Call()
        fork = ProcessFork(f, g, Cls, call, Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual(
            (f, g, Cls, call, Cls.c, cls.m, cls.s),
            fork.calls
        )
        self.default_attributes(fork)

    def test_list(self):
        fork = ProcessFork([f, g])
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g), fork.calls)
        self.default_attributes(fork)

    def test_list_and_calls(self):
        call = Call()
        fork = ProcessFork([f, g], Cls, call)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g, Cls, call), fork.calls)
        self.default_attributes(fork)

    def test_decorator(self):

        @ProcessFork
        def func(x):
            return x

        self.assertIsInstance(func, ProcessFork)
        self.assertTrue(hasattr(func, 'calls'))
        self.assertIsInstance(func.calls, tuple)
        self.assertEqual(1, len(func.calls))
        self.default_attributes(func)

    def test_non_callable_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ProcessFork(cls)

    def test_non_callables_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ProcessFork([f, 1, cls, g])

    def test_non_callable_args_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ProcessFork(f, g, cls, 1, Cls)

    def test_processpool_result_attributes(self):
        fork = ProcessFork(
            [f],
            max_workers=8,
            initializer=g,
            initargs=(1, 2),
            max_tasks_per_child=5,
            timeout=42
        )
        self.assertTrue(hasattr(fork, 'max_workers'))
        self.assertIsInstance(fork.max_workers, int)
        self.assertEqual(8, fork.max_workers)
        self.assertTrue(hasattr(fork, 'initializer'))
        self.assertIs(fork.initializer, g)
        self.assertTrue(hasattr(fork, 'initargs'))
        self.assertTupleEqual((1, 2), fork.initargs)
        self.assertTrue(hasattr(fork, 'max_tasks_per_child'))
        self.assertIsInstance(fork.max_tasks_per_child, int)
        self.assertEqual(5, fork.max_tasks_per_child)
        self.assertTrue(hasattr(fork, 'timeout'))
        self.assertTrue(isinstance(fork.timeout, int))
        self.assertEqual(42, fork.timeout)


class TestDefaultUsage(unittest.TestCase):

    def test_callable(self):
        fork = ProcessFork(f, lambda x: x, Cls, Call())
        self.assertTrue(callable(fork))

    def test_empty_no_arg(self):
        fork = ProcessFork()
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_one_arg(self):
        fork = ProcessFork()
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_two_args(self):
        fork = ProcessFork()
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_no_args_no_return_value(self):
        fork = ProcessFork(f)
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_no_args_return_value(self):
        fork = ProcessFork(return_value)
        result = fork()
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_single_no_args_return_values(self):
        fork = ProcessFork(return_values)
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_arg_no_return_value(self):
        fork = ProcessFork(no_return_value)
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_arg_return_value(self):
        fork = ProcessFork(return_value)
        result = fork(2)
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_single_arg_return_values(self):
        fork = ProcessFork(return_values)
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_args_no_return_value(self):
        fork = ProcessFork(no_return_value)
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_args_return_value(self):
        fork = ProcessFork(return_value)
        result = fork('foo', 2)
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_single_args_return_values(self):
        fork = ProcessFork(return_values)
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_multi_no_return_values(self):
        fork = ProcessFork(no_return_value, no_return_value)
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_multi_return_value_first(self):
        fork = ProcessFork(return_value, no_return_value)
        result = fork(1)
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_value_second(self):
        fork = ProcessFork(no_return_value, return_value)
        result = fork(1)
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_values(self):
        fork = ProcessFork(return_value, return_value)
        result = fork(1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 1), result)

    def test_raises(self):
        fork = ProcessFork(no_return_value, g)
        expected = ('\nAttributeError executing\n'
                    'g\n'
                    'in fork 1 of\n'
                    "ProcessFork(4, None, (), None, None):\n"
                    '[ 0] no_return_value\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        fork = ProcessFork(A(1), g)
        expected = ('\nAttributeError executing\n'
                    'A(1)\n'
                    'in fork 0 of\n'
                    "ProcessFork(4, None, (), None, None):\n"
                    '[ 0] A(1)\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        fork = ProcessFork(Ind([1]), g)
        expected = ('\nAttributeError executing\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    'in fork 0 of\n'
                    "ProcessFork(4, None, (), None, None):\n"
                    '[ 0] Ind():\n'
                    '     [ 0] 1\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))


class TestProcessPoolUsage(unittest.TestCase):

    @patch('swak.funcflow.concurrent.processfork.ProcessPoolExecutor')
    def test_processpool_called(self, cls):
        fork = ProcessFork(
            h,
            max_workers=8,
            initializer=h,
            initargs=(1, 2),
            max_tasks_per_child=5
        )
        _ = fork(3, 4)
        cls.assert_called_once()

    @patch('swak.funcflow.concurrent.processfork.ProcessPoolExecutor')
    def test_processpool_called_with_processpoolargs(self, cls):
        fork = ProcessFork(
            h,
            max_workers=8,
            initializer=h,
            initargs=(1, 2),
            max_tasks_per_child=5
        )
        _ = fork(3, 4)
        cls.assert_called_once_with(8, None, h, (1, 2), max_tasks_per_child=5)


class TestResultUsage(unittest.TestCase):

    @patch('concurrent.futures.Future.result')
    def test_result_called_once(self, method):
        fork = ProcessFork(h)
        _ = fork(1, 2)
        method.assert_called_once()

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice(self, method):
        fork = ProcessFork([h], h)
        _ = fork(1, 2)
        self.assertEqual(2, method.call_count)

    @patch('concurrent.futures.Future.result')
    def test_result_called_once_no_timeout(self, method):
        fork = ProcessFork(h)
        _ = fork(1, 2)
        method.assert_called_once_with(None)

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice_no_timeout(self, method):
        fork = ProcessFork([h], h)
        _ = fork(1, 2)
        ((a,), _), ((b,), _) = method.call_args_list
        self.assertIsNone(a)
        self.assertIsNone(b)

    @patch('concurrent.futures.Future.result')
    def test_result_called_once_timeout(self, method):
        fork = ProcessFork(h, timeout=42)
        _ = fork(1, 2)
        method.assert_called_once_with(42)

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice_timeout(self, method):
        fork = ProcessFork([h], h, timeout=42)
        _ = fork(1, 2)
        ((a,), _), ((b,), _) = method.call_args_list
        self.assertIsInstance(a, int)
        self.assertEqual(42, a)
        self.assertIsInstance(b, int)
        self.assertEqual(42, b)


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = f, lambda x: x, Cls, Call()
        self.fork = ProcessFork(*self.calls)

    def test_iter(self):
        for i, call in enumerate(self.fork):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(4, len(self.fork))
        self.assertEqual(0, len(ProcessFork()))

    def test_bool(self):
        self.assertFalse(ProcessFork())
        self.assertTrue(self.fork)

    def test_contains(self):
        self.assertIn(Cls, self.fork)

    def test_reversed(self):
        self.assertIsInstance(reversed(self.fork), ProcessFork)
        expected = list(reversed(self.calls))
        for i, call in enumerate(reversed(self.fork)):
            self.assertIs(expected[i], call)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.fork[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.fork[:1], ProcessFork)
        self.assertTupleEqual(self.calls[:1], self.fork[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.fork[:3], ProcessFork)
        self.assertTupleEqual(self.calls[:3], self.fork[:3].calls)

    def test_equality_true_self(self):
        self.assertEqual(self.fork, self.fork)

    def test_equality_true_other(self):
        self.assertEqual(self.fork, ProcessFork(*self.calls))

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.fork == 'foo')

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.fork == ProcessFork())

    def test_inequality_false_self(self):
        self.assertFalse(self.fork != self.fork)

    def test_inequality_false_other(self):
        self.assertFalse(self.fork != ProcessFork(*self.calls))

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.fork, 'foo')

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.fork, ProcessFork())

    def test_add_call(self):
        fork = self.fork + f
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual((*self.calls, f), fork.calls)

    def test_add_empty_calls(self):
        fork = self.fork + []
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_calls(self):
        fork = self.fork + [f, g]
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual((*self.calls, f, g), fork.calls)

    def test_add_empty_self(self):
        fork = self.fork + ProcessFork()
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_self(self):
        fork = self.fork + ProcessFork(f, g)
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual((*self.calls, f, g), fork.calls)

    def test_add_non_callable_raises(self):
        cls = Cls()
        with self.assertRaises(TypeError):
            _ = self.fork + cls

    def test_add_non_callables_raises(self):
        cls = Cls()
        with self.assertRaises(TypeError):
            _ = self.fork + [f, cls, 1, g]

    def test_radd_call(self):
        fork = f + self.fork
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual((f, *self.calls), fork.calls)

    def test_radd_empty_calls(self):
        fork = [] + self.fork
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_radd_calls(self):
        fork = [f, g] + self.fork
        self.assertIsInstance(fork, ProcessFork)
        self.assertTupleEqual((f, g, *self.calls), fork.calls)

    def test_radd_non_callable_raises(self):
        cls = Cls()
        with self.assertRaises(TypeError):
            _ = cls + self.fork

    def test_radd_non_callables_raises(self):
        cls = Cls()
        with self.assertRaises(TypeError):
            _ = [f, cls, 1, g] + self.fork


class TestMisc(unittest.TestCase):

    def test_pickle_works(self):
        fork = ProcessFork(f, Cls, Call())
        _ = pickle.dumps(fork)

    def test_pickle_raises_with_lambdas(self):
        fork = ProcessFork(f, Cls, Call(), lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(fork)

    def test_type_annotation(self):
        _ = ProcessFork[[int, bool, str], float](f, Cls)

    def test_type_annotation_tuple(self):
        _ = ProcessFork[[int, bool, str], tuple[float, dict]](f, Cls)

    def test_flat(self):
        fork = ProcessFork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        expected = (
            "ProcessFork(4, None, (), None, None):\n"
            "[ 0] lambda\n"
            "[ 1] f\n"
            "[ 2] Cls\n"
            "[ 3] Cls.c\n"
            "[ 4] Cls.m\n"
            "[ 5] Cls.s\n"
            "[ 6] Call(...)\n"
            "[ 7] A('foo')"
        )
        self.assertEqual(expected, repr(fork))

    def test_nested(self):
        fork = ProcessFork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = ProcessFork(fork, fork)
        expected = (
            "ProcessFork(4, None, (), None, None):\n"
            "[ 0] ProcessFork(4, None, (), None, None):\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] ProcessFork(4, None, (), None, None):\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')"
        )
        self.assertEqual(expected, repr(outer))

    def test_attribute_repr(self):
        fork = ProcessFork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo'),
            max_workers=8,
            initializer=g,
            initargs=(1, 2),
            max_tasks_per_child=5,
            timeout=42
        )
        expected = (
            "ProcessFork(8, g, (1, 2), 5, 42):\n"
            "[ 0] lambda\n"
            "[ 1] f\n"
            "[ 2] Cls\n"
            "[ 3] Cls.c\n"
            "[ 4] Cls.m\n"
            "[ 5] Cls.s\n"
            "[ 6] Call(...)\n"
            "[ 7] A('foo')"
        )
        self.assertEqual(expected, repr(fork))

    def test_empty_repr(self):
        fork = ProcessFork()
        expected = "ProcessFork(4, None, (), None, None)"
        self.assertEqual(expected, repr(fork))

    def test_empty_attribute_repr(self):
        fork = ProcessFork(
            max_workers=8,
            initializer=g,
            initargs=(1, 2),
            max_tasks_per_child=5,
            timeout=42
        )
        expected = "ProcessFork(8, g, (1, 2), 5, 42)"
        self.assertEqual(expected, repr(fork))


if __name__ == '__main__':
    unittest.main()

import unittest
import pickle
from unittest.mock import Mock, patch
from swak.funcflow.concurrent import ThreadFork
from swak.funcflow.exceptions import ForkError
from swak.misc import ArgRepr, IndentRepr


def f():
    return ()


def g(*_):
    raise AttributeError('Test!')


def h(x, y):
    return x + y


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
        self.assertTrue(hasattr(fork, 'thread_name_prefix'))
        self.assertIsInstance(fork.thread_name_prefix, str)
        self.assertTrue(hasattr(fork, 'initializer'))
        self.assertIsNone(fork.initializer)
        self.assertTrue(hasattr(fork, 'initargs'))
        self.assertTupleEqual((), fork.initargs)
        self.assertTrue(hasattr(fork, 'timeout'))
        self.assertIsNone(fork.timeout)

    def test_empty(self):
        fork = ThreadFork()
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((), fork.calls)
        self.default_attributes(fork)

    def test_empty_list(self):
        fork = ThreadFork([])
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((), fork.calls)
        self.default_attributes(fork)

    def test_function(self):
        fork = ThreadFork(f)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, ), fork.calls)
        self.default_attributes(fork)

    def test_functions(self):
        fork = ThreadFork(f, g)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g), fork.calls)
        self.default_attributes(fork)

    def test_lambda(self):
        fork = ThreadFork(lambda x: x)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertEqual(1, len(fork.calls))
        self.default_attributes(fork)

    def test_lambdas(self):
        fork = ThreadFork(lambda x: x, lambda y: ())
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertEqual(2, len(fork.calls))
        self.default_attributes(fork)

    def test_class(self):
        fork = ThreadFork(Cls)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls,), fork.calls)
        self.default_attributes(fork)

    def test_classes(self):
        fork = ThreadFork(Cls, Call, A)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls, Call, A), fork.calls)
        self.default_attributes(fork)

    def test_object(self):
        call = Call()
        fork = ThreadFork(call)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((call,), fork.calls)
        self.default_attributes(fork)

    def test_method(self):
        cls = Cls()
        fork = ThreadFork(cls.m)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((cls.m,), fork.calls)
        self.default_attributes(fork)

    def test_methods(self):
        cls = Cls()
        fork = ThreadFork(Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((Cls.c, cls.m, cls.s), fork.calls)
        self.default_attributes(fork)

    def test_mix(self):
        cls = Cls()
        call = Call()
        fork = ThreadFork(f, g, Cls, call, Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual(
            (f, g, Cls, call, Cls.c, cls.m, cls.s),
            fork.calls
        )
        self.default_attributes(fork)

    def test_list(self):
        fork = ThreadFork([f, g])
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g), fork.calls)
        self.default_attributes(fork)

    def test_list_and_calls(self):
        call = Call()
        fork = ThreadFork([f, g], Cls, call)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, g, Cls, call), fork.calls)
        self.default_attributes(fork)

    def test_decorator(self):

        @ThreadFork
        def h(x):
            return x

        self.assertIsInstance(h, ThreadFork)
        self.assertTrue(hasattr(h, 'calls'))
        self.assertIsInstance(h.calls, tuple)
        self.assertEqual(1, len(h.calls))
        self.default_attributes(h)

    def test_non_callable_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ThreadFork(cls)

    def test_non_callables_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ThreadFork([f, 1, cls, g])

    def test_non_callable_args_raises(self):
        cls = Cls()
        with self.assertRaises(ForkError):
            _ = ThreadFork(f, g, cls, 1, Cls)

    def test_threadpool_result_attributes(self):
        fork = ThreadFork(
            [f],
            max_workers=8,
            thread_name_prefix='tn',
            initializer=g,
            initargs=(1, 2),
            timeout=42
        )
        self.assertTrue(hasattr(fork, 'max_workers'))
        self.assertIsInstance(fork.max_workers, int)
        self.assertEqual(8, fork.max_workers)
        self.assertTrue(hasattr(fork, 'thread_name_prefix'))
        self.assertEqual('tn', fork.thread_name_prefix)
        self.assertTrue(hasattr(fork, 'initializer'))
        self.assertIs(fork.initializer, g)
        self.assertTrue(hasattr(fork, 'initargs'))
        self.assertTupleEqual((1, 2), fork.initargs)
        self.assertTrue(hasattr(fork, 'timeout'))
        self.assertTrue(isinstance(fork.timeout, int))
        self.assertEqual(42, fork.timeout)


class TestDefaultUsage(unittest.TestCase):

    def test_callable(self):
        fork = ThreadFork(f, lambda x: x, Cls, Call())
        self.assertTrue(callable(fork))

    def test_empty_no_arg(self):
        fork = ThreadFork()
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_one_arg(self):
        fork = ThreadFork()
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_two_args(self):
        fork = ThreadFork()
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_calls_called_with_arg(self):
        mock_1 = Mock()
        mock_2 = Mock()
        mock_3 = Mock()
        fork = ThreadFork(mock_1, mock_2, mock_3)
        _ = fork('foo')
        mock_1.assert_called_once()
        mock_1.assert_called_once_with('foo')
        mock_2.assert_called_once()
        mock_2.assert_called_once_with('foo')
        mock_3.assert_called_once()
        mock_3.assert_called_once_with('foo')

    def test_single_no_args_no_return_value(self):
        fork = ThreadFork(lambda: ())
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_no_args_return_value(self):
        fork = ThreadFork(lambda: 2)
        result = fork()
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_no_args_return_values(self):
        fork = ThreadFork(lambda: (1, 2))
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_arg_no_return_value(self):
        fork = ThreadFork(lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_arg_return_value(self):
        fork = ThreadFork(lambda x: 2)
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_arg_return_values(self):
        fork = ThreadFork(lambda x: (1, 2))
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_args_no_return_value(self):
        fork = ThreadFork(lambda *x: ())
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_args_return_value(self):
        fork = ThreadFork(lambda *x: 2)
        result = fork('foo', 1)
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_args_return_values(self):
        fork = ThreadFork(lambda *x: (1, 2))
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_multi_no_return_values(self):
        fork = ThreadFork(lambda x: (), lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_multi_return_value_first(self):
        fork = ThreadFork(lambda x: 1, lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_value_second(self):
        fork = ThreadFork(lambda x: (), lambda x: 1)
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_values(self):
        fork = ThreadFork(lambda x: 1, lambda x: 2)
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_raises(self):
        fork = ThreadFork(lambda *x: x, g)
        expected = ('\nAttributeError executing\n'
                    'g\n'
                    'in fork 1 of\n'
                    "ThreadFork(16, '', None, (), None):\n"
                    '[ 0] lambda\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        fork = ThreadFork(A(1), g)
        expected = ('\nAttributeError executing\n'
                    'A(1)\n'
                    'in fork 0 of\n'
                    "ThreadFork(16, '', None, (), None):\n"
                    '[ 0] A(1)\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        fork = ThreadFork(Ind([1]), g)
        expected = ('\nAttributeError executing\n'
                    'Ind():\n'
                    '[ 0] 1\n'
                    'in fork 0 of\n'
                    "ThreadFork(16, '', None, (), None):\n"
                    '[ 0] Ind():\n'
                    '     [ 0] 1\n'
                    '[ 1] g\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))


class TestThreadPoolUsage(unittest.TestCase):

    @patch('swak.funcflow.concurrent.threadfork.ThreadPoolExecutor')
    def test_threadpool_called(self, cls):
        fork = ThreadFork(
            h,
            max_workers=8,
            thread_name_prefix='tn',
            initializer=h,
            initargs=(1, 2)
        )
        _ = fork(3, 4)
        cls.assert_called_once()

    @patch('swak.funcflow.concurrent.threadfork.ThreadPoolExecutor')
    def test_threadpool_called_with_threadpoolargs(self, cls):
        fork = ThreadFork(
            h,
            max_workers=8,
            thread_name_prefix='tn',
            initializer=h,
            initargs=(1, 2)
        )
        _ = fork(3, 4)
        cls.assert_called_once_with(8, 'tn', h, (1, 2))


class TestResultUsage(unittest.TestCase):

    @patch('concurrent.futures.Future.result')
    def test_result_called_once(self, method):
        fork = ThreadFork(h)
        _ = fork(1, 2)
        method.assert_called_once()

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice(self, method):
        fork = ThreadFork([h], h)
        _ = fork(1, 2)
        self.assertEqual(2, method.call_count)

    @patch('concurrent.futures.Future.result')
    def test_result_called_once_no_timeout(self, method):
        fork = ThreadFork(h)
        _ = fork(1, 2)
        method.assert_called_once_with(None)

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice_no_timeout(self, method):
        fork = ThreadFork([h], h)
        _ = fork(1, 2)
        ((a,), _), ((b,), _) = method.call_args_list
        self.assertIsNone(a)
        self.assertIsNone(b)

    @patch('concurrent.futures.Future.result')
    def test_result_called_once_timeout(self, method):
        fork = ThreadFork(h, timeout=42)
        _ = fork(1, 2)
        method.assert_called_once_with(42)

    @patch('concurrent.futures.Future.result')
    def test_result_called_twice_timeout(self, method):
        fork = ThreadFork([h], h, timeout=42)
        _ = fork(1, 2)
        ((a,), _), ((b,), _) = method.call_args_list
        self.assertIsInstance(a, int)
        self.assertEqual(42, a)
        self.assertIsInstance(b, int)
        self.assertEqual(42, b)


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = f, lambda x: x, Cls, Call()
        self.fork = ThreadFork(*self.calls)

    def test_iter(self):
        for i, call in enumerate(self.fork):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(4, len(self.fork))
        self.assertEqual(0, len(ThreadFork()))

    def test_bool(self):
        self.assertFalse(ThreadFork())
        self.assertTrue(self.fork)

    def test_contains(self):
        self.assertIn(Cls, self.fork)

    def test_reversed(self):
        self.assertIsInstance(reversed(self.fork), ThreadFork)
        expected = list(reversed(self.calls))
        for i, call in enumerate(reversed(self.fork)):
            self.assertIs(expected[i], call)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.fork[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.fork[:1], ThreadFork)
        self.assertTupleEqual(self.calls[:1], self.fork[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.fork[:3], ThreadFork)
        self.assertTupleEqual(self.calls[:3], self.fork[:3].calls)

    def test_equality_true_self(self):
        self.assertEqual(self.fork, self.fork)

    def test_equality_true_other(self):
        self.assertEqual(self.fork, ThreadFork(*self.calls))

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.fork == 'foo')

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.fork == ThreadFork())

    def test_inequality_false_self(self):
        self.assertFalse(self.fork != self.fork)

    def test_inequality_false_other(self):
        self.assertFalse(self.fork != ThreadFork(*self.calls))

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.fork, 'foo')

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.fork, ThreadFork())

    def test_add_call(self):
        fork = self.fork + f
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual((*self.calls, f), fork.calls)

    def test_add_empty_calls(self):
        fork = self.fork + []
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_calls(self):
        fork = self.fork + [f, g]
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual((*self.calls, f, g), fork.calls)

    def test_add_empty_self(self):
        fork = self.fork + ThreadFork()
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_self(self):
        fork = self.fork + ThreadFork(f, g)
        self.assertIsInstance(fork, ThreadFork)
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
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual((f, *self.calls), fork.calls)

    def test_radd_empty_calls(self):
        fork = [] + self.fork
        self.assertIsInstance(fork, ThreadFork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_radd_calls(self):
        fork = [f, g] + self.fork
        self.assertIsInstance(fork, ThreadFork)
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
        fork = ThreadFork(f, Cls, Call())
        _ = pickle.dumps(fork)

    def test_pickle_raises_with_lambdas(self):
        fork = ThreadFork(f, Cls, Call(), lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(fork)

    def test_type_annotation(self):
        _ = ThreadFork[[int, bool, str], float](f, Cls)

    def test_type_annotation_tuple(self):
        _ = ThreadFork[[int, bool, str], tuple[float, dict]](f, Cls)

    def test_flat(self):
        fork = ThreadFork(
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
            "ThreadFork(16, '', None, (), None):\n"
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
        fork = ThreadFork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = ThreadFork(fork, fork)
        expected = (
            "ThreadFork(16, '', None, (), None):\n"
            "[ 0] ThreadFork(16, '', None, (), None):\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] ThreadFork(16, '', None, (), None):\n"
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
        fork = ThreadFork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo'),
            max_workers=8,
            thread_name_prefix='tn',
            initializer=g,
            initargs=(1, 2),
            timeout=42
        )
        expected = (
            "ThreadFork(8, 'tn', g, (1, 2), 42):\n"
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
        fork = ThreadFork()
        expected = "ThreadFork(16, '', None, (), None)"
        self.assertEqual(expected, repr(fork))

    def test_empty_attribute_repr(self):
        fork = ThreadFork(
            max_workers=8,
            thread_name_prefix='tn',
            initializer=g,
            initargs=(1, 2),
            timeout=42
        )
        expected = "ThreadFork(8, 'tn', g, (1, 2), 42)"
        self.assertEqual(expected, repr(fork))


if __name__ == '__main__':
    unittest.main()

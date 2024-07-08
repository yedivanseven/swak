import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Fork
from swak.funcflow.exceptions import ForkError
from swak.magic import ArgRepr


def f():
    return ()


def g(*x):
    raise AttributeError('Test!')


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

    def __call__(self):
        pass


class TestInstantiation(unittest.TestCase):

    def test_empty(self):
        _ = Fork()

    def test_function(self):
        _ = Fork(f)

    def test_functions(self):
        _ = Fork(f, g)

    def test_lambda(self):
        _ = Fork(lambda x: x)

    def test_lambdas(self):
        _ = Fork(lambda x: x, lambda y: ())

    def test_class(self):
        _ = Fork(Cls)

    def test_classes(self):
        _ = Fork(Cls, Call)

    def test_object(self):
        _ = Fork(Call())

    def test_objects(self):
        _ = Fork(Call(), Call())

    def test_mix(self):
        _ = Fork(f, lambda x: x, Cls, Call())

    def test_decorator(self):

        @Fork
        def h(x):
            return x

        self.assertIsInstance(h, Fork)

    def test_pickle_works(self):
        fork = Fork(f, Cls, Call())
        _ = pickle.dumps(fork)

    def test_pickle_raises_with_lambdas(self):
        fork = Fork(f, Cls, Call(), lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(fork)

    def test_callable(self):
        fork = Fork(f, lambda x: x, Cls, Call())
        self.assertTrue(callable(fork))

    def test_attribute_empty(self):
        fork = Fork()
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((), fork.calls)

    def test_attribute(self):
        call = Call()
        fork = Fork(f, Cls, call)
        self.assertTrue(hasattr(fork, 'calls'))
        self.assertIsInstance(fork.calls, tuple)
        self.assertTupleEqual((f, Cls, call), fork.calls)

    def test_type_annotation(self):
        _ = Fork[[int, bool, str], float](f, Cls)

    def test_type_annotation_tuple(self):
        _ = Fork[[int, bool, str], tuple[float, dict]](f, Cls)


class TestFunctionality(unittest.TestCase):

    def test_empty_no_arg(self):
        fork = Fork()
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_one_arg(self):
        fork = Fork()
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_two_args(self):
        fork = Fork()
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_calls_called_with_arg(self):
        mock_1 = Mock()
        mock_2 = Mock()
        mock_3 = Mock()
        fork = Fork(mock_1, mock_2, mock_3)
        _ = fork('foo')
        mock_1.assert_called_once()
        mock_1.assert_called_once_with('foo')
        mock_2.assert_called_once()
        mock_2.assert_called_once_with('foo')
        mock_3.assert_called_once()
        mock_3.assert_called_once_with('foo')

    def test_single_no_args_no_return_value(self):
        fork = Fork(lambda: ())
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_no_args_return_value(self):
        fork = Fork(lambda: 2)
        result = fork()
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_no_args_return_values(self):
        fork = Fork(lambda: (1, 2))
        result = fork()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_arg_no_return_value(self):
        fork = Fork(lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_arg_return_value(self):
        fork = Fork(lambda x: 2)
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_arg_return_values(self):
        fork = Fork(lambda x: (1, 2))
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_single_args_no_return_value(self):
        fork = Fork(lambda *x: ())
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_single_args_return_value(self):
        fork = Fork(lambda *x: 2)
        result = fork('foo', 1)
        self.assertIsInstance(result, int)
        self.assertEqual(2, result)

    def test_single_args_return_values(self):
        fork = Fork(lambda *x: (1, 2))
        result = fork('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_multi_no_return_values(self):
        fork = Fork(lambda x: (), lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_multi_return_value_first(self):
        fork = Fork(lambda x: 1, lambda x: ())
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_value_second(self):
        fork = Fork(lambda x: (), lambda x: 1)
        result = fork('foo')
        self.assertIsInstance(result, int)
        self.assertEqual(1, result)

    def test_multi_return_values(self):
        fork = Fork(lambda x: 1, lambda x: 2)
        result = fork('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_raises(self):
        fork = Fork(lambda *x: x, g)
        expected = ('Error executing\n'
                    'g\n'
                    'in fork 1 of\n'
                    'Fork:\n'
                    '[ 0] lambda\n'
                    '[ 1] g\n'
                    'AttributeError:\n'
                    'Test!')
        with self.assertRaises(ForkError) as error:
            _ = fork('foo', 1)
        self.assertEqual(expected, str(error.exception))


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = f, lambda x: x, Cls, Call()
        self.fork = Fork(*self.calls)

    def test_iter(self):
        for i, call in enumerate(self.fork):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(4, len(self.fork))
        self.assertEqual(0, len(Fork()))

    def test_bool(self):
        self.assertFalse(Fork())
        self.assertTrue(self.fork)

    def test_contains(self):
        self.assertIn(Cls, self.fork)

    def test_reversed(self):
        self.assertIsInstance(reversed(self.fork), Fork)
        expected = list(reversed(self.calls))
        for i, call in enumerate(reversed(self.fork)):
            self.assertIs(expected[i], call)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.fork[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.fork[:1], Fork)
        self.assertTupleEqual(self.calls[:1], self.fork[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.fork[:3], Fork)
        self.assertTupleEqual(self.calls[:3], self.fork[:3].calls)

    def test_equality_true_self(self):
        self.assertEqual(self.fork, self.fork)

    def test_equality_true_other(self):
        self.assertEqual(self.fork, Fork(*self.calls))

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.fork == 'foo')

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.fork == Fork())

    def test_inequality_false_self(self):
        self.assertFalse(self.fork != self.fork)

    def test_inequality_false_other(self):
        self.assertFalse(self.fork != Fork(*self.calls))

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.fork, 'foo')

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.fork, Fork())

    def test_add_call(self):
        fork = self.fork + f
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual((*self.calls, f), fork.calls)

    def test_add_empty_calls(self):
        fork = self.fork + []
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_calls(self):
        fork = self.fork + [f, g]
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual((*self.calls, f, g), fork.calls)

    def test_add_empty_self(self):
        fork = self.fork + Fork()
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_add_self(self):
        fork = self.fork + Fork(f, g)
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual((*self.calls, f, g), fork.calls)

    def test_radd_call(self):
        fork = f + self.fork
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual((f, *self.calls), fork.calls)

    def test_radd_empty_calls(self):
        fork = [] + self.fork
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual(self.calls, fork.calls)

    def test_radd_calls(self):
        fork = [f, g] + self.fork
        self.assertIsInstance(fork, Fork)
        self.assertTupleEqual((f, g, *self.calls), fork.calls)


class TestRepr(unittest.TestCase):

    def test_flat(self):
        fork = Fork(
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
            "Fork:\n"
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
        fork = Fork(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = Fork(fork, fork)
        expected = (
            "Fork:\n"
            "[ 0] Fork:\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] Fork:\n"
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


if __name__ == '__main__':
    unittest.main()

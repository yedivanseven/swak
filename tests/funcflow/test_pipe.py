import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Pipe
from swak.funcflow.exceptions import PipeError
from swak.magic import ArgRepr, IndentRepr


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

    def __call__(self, *_):
        raise AttributeError('Test!')


class Ind(IndentRepr):

    def __call__(self, *_):
        raise AttributeError('Test!')


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        pipe = Pipe()
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((), pipe.calls)

    def test_empty_list(self):
        pipe = Pipe([])
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((), pipe.calls)

    def test_function(self):
        pipe = Pipe(f)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((f,), pipe.calls)

    def test_functions(self):
        pipe = Pipe(f, g)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((f, g), pipe.calls)

    def test_lambda(self):
        pipe = Pipe(lambda x: x)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)

    def test_lambdas(self):
        pipe = Pipe(lambda x: x, lambda y: ())
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)

    def test_class(self):
        pipe = Pipe(Cls)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((Cls, ), pipe.calls)

    def test_classes(self):
        pipe = Pipe(Cls, Call, A)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((Cls, Call, A), pipe.calls)

    def test_object(self):
        call = Call()
        pipe = Pipe(call)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((call,), pipe.calls)

    def test_objects(self):
        call = Call()
        cls = Cls()
        pipe = Pipe(call, cls)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((call, cls), pipe.calls)

    def test_method(self):
        cls = Cls()
        pipe = Pipe(cls.m)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((cls.m,), pipe.calls)

    def test_methods(self):
        cls = Cls()
        pipe = Pipe(Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((Cls.c, cls.m, cls.s), pipe.calls)

    def test_mix(self):
        cls = Cls()
        call = Call()
        pipe = Pipe(f, g, Cls, call, Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual(
            (f, g, Cls, call, Cls.c, cls.m, cls.s),
            pipe.calls
        )

    def test_list(self):
        pipe = Pipe([f, g])
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((f, g), pipe.calls)

    def test_list_and_calls(self):
        pipe = Pipe([f, g], Cls, Cls.c)
        self.assertTrue(hasattr(pipe, 'calls'))
        self.assertIsInstance(pipe.calls, tuple)
        self.assertTupleEqual((f, g, Cls, Cls.c), pipe.calls)

    def test_decorator(self):

        @Pipe
        def h(x):
            return x

        self.assertIsInstance(h, Pipe)
        self.assertTrue(hasattr(h, 'calls'))
        self.assertIsInstance(h.calls, tuple)
        self.assertEqual(1, len(h.calls))


class TestUsage(unittest.TestCase):

    def test_callable(self):
        pipe = Pipe(f, lambda x: x, Cls, Call())
        self.assertTrue(callable(pipe))

    def test_empty_no_arg(self):
        pipe = Pipe()
        result = pipe()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_one_arg(self):
        pipe = Pipe()
        result = pipe('foo')
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(('foo', ), result)

    def test_empty_two_args(self):
        pipe = Pipe()
        result = pipe('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(('foo', 1), result)

    def test_empty_empty(self):
        pipe = Pipe(lambda: (), lambda: ())
        result = pipe()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_arg_empty(self):
        pipe = Pipe(lambda x: (), lambda: ())
        result = pipe(1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_arg(self):
        pipe = Pipe(lambda: (), lambda: 3)
        result = pipe()
        self.assertIsInstance(result, int)
        self.assertEqual(3, result)

    def test_arg_empty_arg(self):
        pipe = Pipe(lambda x: (), lambda: 3)
        result = pipe(1)
        self.assertIsInstance(result, int)
        self.assertEqual(3, result)

    def test_args_passed_to_first_call(self):
        mock = Mock()
        pipe = Pipe(mock, Mock())
        _ = pipe('foo', 1)
        mock.assert_called_once()
        mock.assert_called_once_with('foo', 1)

    def test_chaining(self):
        a = Mock(return_value='bar')
        b = Mock(return_value='baz')
        c = Mock(return_value=42)
        pipe = Pipe(a, b, c)
        result = pipe('foo', 1)
        self.assertIsInstance(result, int)
        self.assertEqual(42, result)
        a.assert_called_once()
        a.assert_called_once_with('foo', 1)
        b.assert_called_once()
        b.assert_called_once_with('bar')
        c.assert_called_once()
        c.assert_called_once_with('baz')

    def test_empty_chaining(self):
        pipe = Pipe(f, f, f, f)
        result = pipe()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_raises(self):
        expected = ("\nAttributeError executing\n"
                    "g\n"
                    "in step 1 of\n"
                    "Pipe():\n"
                    "[ 0] f\n"
                    "[ 1] g\n"
                    "[ 2] f\n"
                    "Test!")
        pipe = Pipe(f, g, f)
        with self.assertRaises(PipeError) as error:
            _ = pipe()
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ("\nAttributeError executing\n"
                    "A(1)\n"
                    "in step 1 of\n"
                    "Pipe():\n"
                    "[ 0] f\n"
                    "[ 1] A(1)\n"
                    "[ 2] f\n"
                    "Test!")
        pipe = Pipe(f, A(1), f)
        with self.assertRaises(PipeError) as error:
            _ = pipe()
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ("\nAttributeError executing\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "in step 1 of\n"
                    "Pipe():\n"
                    "[ 0] f\n"
                    "[ 1] Ind():\n"
                    "     [ 0] 1\n"
                    "[ 2] f\n"
                    "Test!")
        pipe = Pipe(f, Ind([1]), f)
        with self.assertRaises(PipeError) as error:
            _ = pipe()
        self.assertEqual(expected, str(error.exception))


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = f, lambda x: x, Cls, Call()
        self.pipe = Pipe(*self.calls)

    def test_iter(self):
        for i, call in enumerate(self.pipe):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(4, len(self.pipe))
        self.assertEqual(0, len(Pipe()))

    def test_bool(self):
        self.assertFalse(Pipe())
        self.assertTrue(self.pipe)

    def test_contains(self):
        self.assertIn(Cls, self.pipe)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.pipe[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.pipe[:1], Pipe)
        self.assertTupleEqual(self.calls[:1], self.pipe[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.pipe[:3], Pipe)
        self.assertTupleEqual(self.calls[:3], self.pipe[:3].calls)

    def test_equality_true_self(self):
        self.assertEqual(self.pipe, self.pipe)

    def test_equality_true_other(self):
        self.assertEqual(self.pipe, Pipe(*self.calls))

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.pipe == 'foo')

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.pipe == Pipe())

    def test_inequality_false_self(self):
        self.assertFalse(self.pipe != self.pipe)

    def test_inequality_false_other(self):
        self.assertFalse(self.pipe != Pipe(*self.calls))

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.pipe, 'foo')

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.pipe, Pipe())

    def test_add_call(self):
        pipe = self.pipe + f
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual((*self.calls, f), pipe.calls)

    def test_add_empty_calls(self):
        pipe = self.pipe + []
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual(self.calls, pipe.calls)

    def test_add_calls(self):
        pipe = self.pipe + [f, g]
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual((*self.calls, f, g), pipe.calls)

    def test_add_empty_self(self):
        pipe = self.pipe + Pipe()
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual(self.calls, pipe.calls)

    def test_add_self(self):
        pipe = self.pipe + Pipe(f, g)
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual((*self.calls, f, g), pipe.calls)

    def test_radd_call(self):
        pipe = f + self.pipe
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual((f, *self.calls), pipe.calls)

    def test_radd_empty_calls(self):
        pipe = [] + self.pipe
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual(self.calls, pipe.calls)

    def test_radd_calls(self):
        pipe = [f, g] + self.pipe
        self.assertIsInstance(pipe, Pipe)
        self.assertTupleEqual((f, g, *self.calls), pipe.calls)


class TestRepr(unittest.TestCase):

    def test_type_annotation(self):
        _ = Pipe[[int, bool, str], float](f, Cls)

    def test_type_annotation_tuple(self):
        _ = Pipe[[int, bool, str], tuple[float, dict]](f, Cls)

    def test_pickle_works(self):
        pipe = Pipe(f, Cls, Call())
        _ = pickle.dumps(pipe)

    def test_pickle_raises_with_lambdas(self):
        pipe = Pipe(f, Cls, Call(), lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(pipe)

    def test_flat(self):
        pipe = Pipe(
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
            "Pipe():\n"
            "[ 0] lambda\n"
            "[ 1] f\n"
            "[ 2] Cls\n"
            "[ 3] Cls.c\n"
            "[ 4] Cls.m\n"
            "[ 5] Cls.s\n"
            "[ 6] Call(...)\n"
            "[ 7] A('foo')"
        )
        self.assertEqual(expected, repr(pipe))

    def test_nested(self):
        pipe = Pipe(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = Pipe(pipe, pipe)
        expected = (
            "Pipe():\n"
            "[ 0] Pipe():\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] Pipe():\n"
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

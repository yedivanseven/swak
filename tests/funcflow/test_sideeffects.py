import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import SideEffects
from swak.funcflow.exceptions import SideEffectsError
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
        s = SideEffects()
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((), s.calls)

    def test_empty_list(self):
        s = SideEffects([])
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((), s.calls)

    def test_function(self):
        s = SideEffects(f)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((f,), s.calls)

    def test_functions(self):
        s = SideEffects(f, g)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((f, g), s.calls)

    def test_lambda(self):
        s = SideEffects(lambda x: x)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)

    def test_lambdas(self):
        s = SideEffects(lambda x: x, lambda y: ())
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)

    def test_class(self):
        s = SideEffects(Cls)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((Cls, ), s.calls)

    def test_classes(self):
        s = SideEffects(Cls, Call, A)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((Cls, Call, A), s.calls)

    def test_object(self):
        call = Call()
        s = SideEffects(call)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((call,), s.calls)

    def test_objects(self):
        call = Call()
        cls = Cls()
        s = SideEffects(call, cls)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((call, cls), s.calls)

    def test_method(self):
        cls = Cls()
        s = SideEffects(cls.m)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((cls.m,), s.calls)

    def test_methods(self):
        cls = Cls()
        s = SideEffects(Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((Cls.c, cls.m, cls.s), s.calls)

    def test_mix(self):
        cls = Cls()
        call = Call()
        s = SideEffects(f, g, Cls, call, Cls.c, cls.m, cls.s)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual(
            (f, g, Cls, call, Cls.c, cls.m, cls.s),
            s.calls
        )

    def test_list(self):
        s = SideEffects([f, g])
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((f, g), s.calls)

    def test_list_and_calls(self):
        s = SideEffects([f, g], Cls, Cls.c)
        self.assertTrue(hasattr(s, 'calls'))
        self.assertIsInstance(s.calls, tuple)
        self.assertTupleEqual((f, g, Cls, Cls.c), s.calls)

    def test_decorator(self):

        @SideEffects
        def h(x):
            return x

        self.assertIsInstance(h, SideEffects)
        self.assertTrue(hasattr(h, 'calls'))
        self.assertIsInstance(h.calls, tuple)
        self.assertEqual(1, len(h.calls))


class TestUsage(unittest.TestCase):

    def test_callable(self):
        s = SideEffects(f, lambda x: x, Cls, Call())
        self.assertTrue(callable(s))

    def test_empty_no_arg(self):
        s = SideEffects()
        result = s()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_empty_one_arg(self):
        s = SideEffects()
        result = s(Cls)
        self.assertIs(result, Cls)

    def test_empty_two_args(self):
        s = SideEffects()
        result = s('foo', 1)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(('foo', 1), result)

    def test_empty_empty(self):
        s = SideEffects(lambda: (), lambda: ())
        result = s()
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((), result)

    def test_one_arg(self):
        s = SideEffects(lambda x: (), lambda x: ())
        result = s(Cls)
        self.assertIs(result, Cls)

    def test_two_args(self):
        s = SideEffects(lambda x, y: (), lambda x, y: ())
        result = s(1, 2)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual((1, 2), result)

    def test_args_passed_to_one_call(self):
        mock = Mock()
        s = SideEffects(mock)
        _ = s('foo', 1)
        mock.assert_called_once()
        mock.assert_called_once_with('foo', 1)

    def test_arg_passed_to_calls(self):
        a = Mock()
        b = Mock()
        c = Mock()
        s = SideEffects(a, b, c)
        result = s(Cls)
        self.assertIs(result, Cls)
        a.assert_called_once()
        a.assert_called_once_with(Cls)
        b.assert_called_once()
        b.assert_called_once_with(Cls)
        c.assert_called_once()
        c.assert_called_once_with(Cls)

    def test_args_passed_to_calls(self):
        a = Mock()
        b = Mock()
        c = Mock()
        s = SideEffects(a, b, c)
        result = s('foo', 42)
        self.assertIsInstance(result, tuple)
        self.assertTupleEqual(('foo', 42), result)
        a.assert_called_once()
        a.assert_called_once_with('foo', 42)
        b.assert_called_once()
        b.assert_called_once_with('foo', 42)
        c.assert_called_once()
        c.assert_called_once_with('foo', 42)

    def test_raises(self):
        expected = ("\nAttributeError executing\n"
                    "g\n"
                    "in step 1 of\n"
                    "SideEffects():\n"
                    "[ 0] f\n"
                    "[ 1] g\n"
                    "[ 2] f\n"
                    "Test!")
        s = SideEffects(f, g, f)
        with self.assertRaises(SideEffectsError) as error:
            _ = s()
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        expected = ("\nAttributeError executing\n"
                    "A(1)\n"
                    "in step 1 of\n"
                    "SideEffects():\n"
                    "[ 0] f\n"
                    "[ 1] A(1)\n"
                    "[ 2] f\n"
                    "Test!")
        s = SideEffects(f, A(1), f)
        with self.assertRaises(SideEffectsError) as error:
            _ = s()
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        expected = ("\nAttributeError executing\n"
                    "Ind():\n"
                    "[ 0] 1\n"
                    "in step 1 of\n"
                    "SideEffects():\n"
                    "[ 0] f\n"
                    "[ 1] Ind():\n"
                    "     [ 0] 1\n"
                    "[ 2] f\n"
                    "Test!")
        s = SideEffects(f, Ind([1]), f)
        with self.assertRaises(SideEffectsError) as error:
            _ = s()
        self.assertEqual(expected, str(error.exception))


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.calls = f, lambda x: x, Cls, Call()
        self.s = SideEffects(*self.calls)

    def test_iter(self):
        for i, call in enumerate(self.s):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(4, len(self.s))
        self.assertEqual(0, len(SideEffects()))

    def test_bool(self):
        self.assertFalse(SideEffects())
        self.assertTrue(self.s)

    def test_contains(self):
        self.assertIn(Cls, self.s)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.s[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.s[:1], SideEffects)
        self.assertTupleEqual(self.calls[:1], self.s[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.s[:3], SideEffects)
        self.assertTupleEqual(self.calls[:3], self.s[:3].calls)

    def test_equality_true_self(self):
        self.assertEqual(self.s, self.s)

    def test_equality_true_other(self):
        self.assertEqual(self.s, SideEffects(*self.calls))

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.s == 'foo')

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.s == SideEffects())

    def test_inequality_false_self(self):
        self.assertFalse(self.s != self.s)

    def test_inequality_false_other(self):
        self.assertFalse(self.s != SideEffects(*self.calls))

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.s, 'foo')

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.s, SideEffects())

    def test_add_call(self):
        s = self.s + f
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual((*self.calls, f), s.calls)

    def test_add_empty_calls(self):
        s = self.s + []
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual(self.calls, s.calls)

    def test_add_calls(self):
        s = self.s + [f, g]
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual((*self.calls, f, g), s.calls)

    def test_add_empty_self(self):
        s = self.s + SideEffects()
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual(self.calls, s.calls)

    def test_add_self(self):
        s = self.s + SideEffects(f, g)
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual((*self.calls, f, g), s.calls)

    def test_radd_call(self):
        s = f + self.s
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual((f, *self.calls), s.calls)

    def test_radd_empty_calls(self):
        s = [] + self.s
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual(self.calls, s.calls)

    def test_radd_calls(self):
        s = [f, g] + self.s
        self.assertIsInstance(s, SideEffects)
        self.assertTupleEqual((f, g, *self.calls), s.calls)


class TestMisc(unittest.TestCase):

    def test_type_annotation(self):
        _ = SideEffects[[int]](f, Cls)

    def test_type_annotation_tuple(self):
        _ = SideEffects[[int, bool, str]](f, Cls)

    def test_pickle_works(self):
        s = SideEffects(f, Cls, Call())
        _ = pickle.dumps(s)

    def test_pickle_raises_with_lambdas(self):
        s = SideEffects(f, Cls, Call(), lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(s)

    def test_flat(self):
        s = SideEffects(
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
            "SideEffects():\n"
            "[ 0] lambda\n"
            "[ 1] f\n"
            "[ 2] Cls\n"
            "[ 3] Cls.c\n"
            "[ 4] Cls.m\n"
            "[ 5] Cls.s\n"
            "[ 6] Call(...)\n"
            "[ 7] A('foo')"
        )
        self.assertEqual(expected, repr(s))

    def test_nested(self):
        s = SideEffects(
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = SideEffects(s, s)
        expected = (
            "SideEffects():\n"
            "[ 0] SideEffects():\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] SideEffects():\n"
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

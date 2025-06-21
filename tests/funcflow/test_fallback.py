import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Fallback
from swak.funcflow.exceptions import FallbackErrors
from swak.funcflow.misc import unit


ERROR = ValueError('Test value error')


def f(*_):
    raise ERROR


def g(x):
    return x + 1,


def h(x: int) -> tuple[int, int]:
    return x, x+2


def q(_):
    return ()


def p(x: int, s: str) -> bool:
    return len(s) > x


def cb(name, args, error):
    return name, args, error


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.empty = Fallback([])
        self.one = Fallback(f)
        self.many = Fallback([f, g])

    def test_empty_has_calls(self):
        self.assertTrue(hasattr(self.empty, 'calls'))

    def test_empty_calls(self):
        self.assertTupleEqual((), self.empty.calls)

    def test_one_has_calls(self):
        self.assertTrue(hasattr(self.one, 'calls'))

    def test_one_calls(self):
        self.assertTupleEqual((f,), self.one.calls)

    def test_many_has_calls(self):
        self.assertTrue(hasattr(self.many, 'calls'))

    def test_many_calls(self):
        self.assertTupleEqual((f, g), self.many.calls)

    def test_has_errors(self):
        self.assertTrue(hasattr(self.one, 'errors'))

    def test_errors(self):
        self.assertSetEqual({Exception}, set(self.one.errors))

    def test_has_callback(self):
        self.assertTrue(hasattr(self.one, 'callback'))

    def test_callback(self):
        self.assertIs(self.one.callback, unit)

    def test_not_callable_raises(self):
        with self.assertRaises(TypeError):
            _ = Fallback(42)

    def test_iterable_not_callable_raises(self):
        with self.assertRaises(TypeError):
            _ = Fallback([f, 42, g])


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.fallback = Fallback(f, TypeError, ValueError, callback=cb)

    def test_errors(self):
        self.assertSetEqual({TypeError, ValueError}, set(self.fallback.errors))

    def test_errors_deduplicated(self):
        fallback = Fallback(f, TypeError, TypeError, callback=cb)
        self.assertTupleEqual((TypeError, ), fallback.errors)

    def test_wrong_errors_raise(self):
        with self.assertRaises(FallbackErrors):
            _ = Fallback(f, TypeError, 'foo', 42, ValueError)

    def test_callback(self):
        self.assertIs(self.fallback.callback, cb)

    def test_wrong_callback_raises(self):
        with self.assertRaises(TypeError):
            _ = Fallback(f, TypeError, ValueError, callback='foo')


class TestDefaultUsage(unittest.TestCase):

    def test_callable(self):
        fallback = Fallback([])
        self.assertTrue(callable(fallback))

    def test_empty_no_args(self):
        callback = Mock()
        empty = Fallback([], callback=callback)
        expected = ()
        actual = empty()
        self.assertTupleEqual(expected, actual)
        callback.assert_not_called()

    def test_empty_one_arg(self):
        callback = Mock()
        empty = Fallback([], callback=callback)
        expected = object()
        actual = empty(expected)
        self.assertIs(expected, actual)
        callback.assert_not_called()

    def test_empty_one_tuple_arg(self):
        callback = Mock()
        empty = Fallback([], callback=callback)
        expected = object(),
        actual = empty(expected)
        self.assertTupleEqual(expected, actual)
        callback.assert_not_called()

    def test_empty_two_tuple_arg(self):
        callback = Mock()
        empty = Fallback([], callback=callback)
        expected = object(), object()
        actual = empty(expected)
        self.assertTupleEqual(expected, actual)
        callback.assert_not_called()

    def test_empty_two_args(self):
        callback = Mock()
        empty = Fallback([], callback=callback)
        expected = object(), object()
        actual = empty(*expected)
        self.assertTupleEqual(expected, actual)
        callback.assert_not_called()

    def test_one_call_called_with_no_args(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one()
        mock.assert_called_once_with()
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_called_with_one_arg(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one(2)
        mock.assert_called_once_with(2)
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_called_with_two_args(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one(2, 'foo')
        mock.assert_called_once_with(2, 'foo')
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_called_with_empty_tuple(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one(())
        mock.assert_called_once_with(())
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_called_with_one_tuple(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one((42,))
        mock.assert_called_once_with((42,))
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_called_with_two_tuple(self):
        callback = Mock()
        mock = Mock(return_value=3)
        one = Fallback(mock, callback=callback)
        actual = one((42, 'foo'))
        mock.assert_called_once_with((42, 'foo'))
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_no_return_value(self):
        callback = Mock()
        one = Fallback(q, callback=callback)
        actual = one(2)
        self.assertTupleEqual((), actual)
        callback.assert_not_called()

    def test_one_call_one_return_value(self):
        callback = Mock()
        one = Fallback(lambda x: x + 1, callback=callback)
        actual = one(2)
        self.assertIsInstance(actual, int)
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_returns_value_from_one_tuple(self):
        callback = Mock()
        one = Fallback(g, callback=callback)
        actual = one(2)
        self.assertIsInstance(actual, int)
        self.assertEqual(3, actual)
        callback.assert_not_called()

    def test_one_call_returns_one_tuple_from_one_tuple(self):
        callback = Mock()
        one = Fallback(lambda x: ((x,),), callback=callback)
        actual = one(2)
        self.assertTupleEqual((2,), actual)
        callback.assert_not_called()

    def test_one_call_returns_two_tuple_from_two_tuple(self):
        callback = Mock()
        one = Fallback(h, callback=callback)
        actual = one(2)
        expected = h(2)
        self.assertTupleEqual(expected, actual)
        callback.assert_not_called()

    def test_one_call_no_arg_raises(self):
        callback = Mock()
        one = Fallback(f, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = one()
        callback.assert_called_once_with('f', (), ERROR)

    def test_one_call_one_arg_raises(self):
        callback = Mock()
        one = Fallback(f, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = one(2)
        callback.assert_called_once_with('f', (2, ), ERROR)

    def test_one_call_two_args_raises(self):
        callback = Mock()
        one = Fallback(f, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = one(2, 3)
        callback.assert_called_once_with('f', (2, 3), ERROR)

    def test_two_calls(self):
        callback = Mock()
        two = Fallback([f, g], callback=callback)
        actual = two(3)
        self.assertIsInstance(actual, int)
        self.assertEqual(4, actual)
        callback.assert_called_once_with('f', (3,), ERROR)

    def test_two_calls_raise(self):
        callback = Mock()
        two = Fallback([f, f], callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = two(3)
        self.assertEqual(2, callback.call_count)
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[0][0]
        )
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[1][0]
        )


class TestErrorsUsage(unittest.TestCase):

    def test_one_call_one_error_raised(self):
        callback = Mock()
        one = Fallback(f, TypeError, callback=callback)
        with self.assertRaises(ValueError):
            _ = one(2)
        callback.assert_not_called()

    def test_one_call_one_error_caught(self):
        callback = Mock()
        one = Fallback(f, ValueError, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = one(2)
        callback.assert_called_once_with('f', (2, ), ERROR)

    def test_one_call_two_errors_raised(self):
        callback = Mock()
        one = Fallback(f, TypeError, AttributeError, callback=callback)
        with self.assertRaises(ValueError):
            _ = one(2)
        callback.assert_not_called()

    def test_one_call_two_errors_caught(self):
        callback = Mock()
        one = Fallback(f, AttributeError, ValueError, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = one(2)
        callback.assert_called_once_with('f', (2, ), ERROR)

    def test_two_calls_one_error_raised(self):
        callback = Mock()
        two = Fallback([f, f], TypeError, callback=callback)
        with self.assertRaises(ValueError):
            _ = two(3)
        callback.assert_not_called()

    def test_two_calls_one_error_caught(self):
        callback = Mock()
        two = Fallback([f, f], ValueError, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = two(3)
        self.assertEqual(2, callback.call_count)
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[0][0]
        )
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[1][0]
        )

    def test_two_calls_two_errors_raised(self):
        callback = Mock()
        two = Fallback([f, f], TypeError, AttributeError, callback=callback)
        with self.assertRaises(ValueError):
            _ = two(3)
        callback.assert_not_called()

    def test_two_calls_two_errors_caught(self):
        callback = Mock()
        two = Fallback([f, f], AttributeError, ValueError, callback=callback)
        with self.assertRaises(FallbackErrors):
            _ = two(3)
        self.assertEqual(2, callback.call_count)
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[0][0]
        )
        self.assertTupleEqual(
            ('f', (3,), ERROR), callback.call_args_list[1][0]
        )


class TestMagic(unittest.TestCase):

    def setUp(self):
        self.calls = f, g, h, q
        self.fallback = Fallback(
            self.calls,
            TypeError,
            ValueError,
            callback=cb
        )

    def test_iter(self):
        print(repr(self.fallback))
        for i, call in enumerate(self.fallback):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(len(self.calls), len(self.fallback))
        self.assertEqual(0, len(Fallback([])))

    def test_bool(self):
        self.assertTrue(self.fallback)
        self.assertFalse(Fallback([]))

    def test_contains(self):
        for call in self.calls:
            self.assertIn(call, self.fallback)

    def test_reversed_raises(self):
        with self.assertRaises(TypeError):
            _ = reversed(self.fallback)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.fallback[i])

    def test_getitem_single_slice(self):
        self.assertIsInstance(self.fallback[:1], Fallback)
        self.assertTupleEqual(self.calls[:1], self.fallback[:1].calls)

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.fallback[:3], Fallback)
        self.assertTupleEqual(self.calls[:3], self.fallback[:3].calls)

    def test_hash(self):
        expected = hash((
            self.fallback.calls,
            self.fallback.errors,
            self.fallback.callback
        ))
        self.assertEqual(expected, hash(self.fallback))

    def test_hash_contract(self):
        fallback = Fallback(self.calls, TypeError, ValueError, callback=cb)
        self.assertEqual(hash(self.fallback), hash(fallback))

    def test_equality_true_self(self):
        self.assertEqual(self.fallback, self.fallback)

    def test_equality_true_other(self):
        fallback = Fallback(self.calls, TypeError, ValueError, callback=cb)
        self.assertEqual(self.fallback, fallback)

    def test_equality_true_ordering_errors(self):
        fallback = Fallback(self.calls, ValueError, TypeError, callback=cb)
        self.assertEqual(self.fallback, fallback)

    def test_equality_false_wrong_class(self):
        self.assertFalse(self.fallback == 'foo')

    def test_equality_false_wrong_calls(self):
        fallback = Fallback(self.calls[:2], TypeError, ValueError, callback=cb)
        self.assertFalse(self.fallback == fallback)

    def test_equality_false_wrong_errors(self):
        fallback = Fallback(self.calls, TypeError, callback=cb)
        self.assertFalse(self.fallback == fallback)

    def test_equality_false_wrong_callback(self):
        fallback = Fallback(self.calls, TypeError, ValueError)
        self.assertFalse(self.fallback == fallback)

    def test_inequality_false_self(self):
        self.assertFalse(self.fallback != self.fallback)

    def test_inequality_false_other(self):
        fallback = Fallback(self.calls, ValueError, TypeError, callback=cb)
        self.assertFalse(self.fallback != fallback)

    def test_inequality_true_wrong_class(self):
        self.assertNotEqual(self.fallback, 'foo')

    def test_inequality_true_wrong_calls(self):
        fallback = Fallback(self.calls[:2], TypeError, ValueError, callback=cb)
        self.assertNotEqual(self.fallback, fallback)

    def test_inequality_true_wrong_errors(self):
        fallback = Fallback(self.calls[:2], ValueError, callback=cb)
        self.assertNotEqual(self.fallback, fallback)

    def test_inequality_true_wrong_callback(self):
        fallback = Fallback(self.calls[:2], TypeError, ValueError)
        self.assertNotEqual(self.fallback, fallback)

    # ToDo: Test also errors and callback!
    def test_add_call(self):
        fallback = self.fallback + f
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual((*self.calls, f), fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_add_empty_calls(self):
        fallback = self.fallback + []
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual(self.calls, fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_add_calls(self):
        fallback = self.fallback + [f, g]
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual((*self.calls, f, g), fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_add_empty_self(self):
        fallback = self.fallback + Fallback([])
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual(self.calls, fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_add_self(self):
        fallback = self.fallback + Fallback([f, g], AttributeError)
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual((*self.calls, f, g), fallback.calls)
        self.assertSetEqual(
            {*self.fallback.errors, AttributeError}, set(fallback.errors)
        )

    def test_add_non_callable_raises(self):
        with self.assertRaises(TypeError):
            _ = self.fallback + 'foo'

    def test_add_non_callables_raises(self):
        with self.assertRaises(TypeError):
            _ = self.fallback + [f, object(), 1, g]

    def test_radd_call(self):
        fallback = f + self.fallback
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual((f, *self.calls), fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_radd_empty_calls(self):
        fallback = [] + self.fallback
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual(self.calls, fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_radd_calls(self):
        fallback = [f, g] + self.fallback
        self.assertIsInstance(fallback, Fallback)
        self.assertTupleEqual((f, g, *self.calls), fallback.calls)
        self.assertTupleEqual(self.fallback.errors, fallback.errors)

    def test_radd_non_callable_raises(self):
        with self.assertRaises(TypeError):
            _ = 'foo' + self.fallback

    def test_radd_non_callables_raises(self):
        with self.assertRaises(TypeError):
            _ = [f, object(), 1, g] + self.fallback


class TestMisc(unittest.TestCase):

    def test_type_annotation_works(self):
        _ = Fallback[[int, str], bool](p)

    def test_empty_pickle_works(self):
        fallback = Fallback([])
        _ = pickle.loads(pickle.dumps(fallback))

    def test_named_call_pickle_works(self):
        fallback = Fallback([f, g], callback=cb)
        _ = pickle.loads(pickle.dumps(fallback))

    def test_lambda_call_pickle_fails(self):
        fallback = Fallback(lambda x: x, callback=cb)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(fallback))

    def test_lambda_callback_pickle_fails(self):
        fallback = Fallback(f, callback=lambda *xs: xs)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(fallback))

    def test_empty_default_repr(self):
        fallback = Fallback([])
        expected = 'Fallback(Exception, callback=unit)'
        self.assertEqual(expected, repr(fallback))

    def test_empty_repr(self):
        fallback = Fallback([], TypeError, ValueError, callback=cb)
        expected = 'Fallback(ValueError, TypeError, callback=cb)'
        self.assertEqual(expected, repr(fallback))

    def test_one_call_default_repr(self):
        fallback = Fallback(f)
        expected = 'Fallback(Exception, callback=unit):\n[ 0] f'
        self.assertEqual(expected, repr(fallback))

    def test_one_call_repr(self):
        fallback = Fallback(f, TypeError, ValueError, callback=cb)
        expected = 'Fallback(ValueError, TypeError, callback=cb):\n[ 0] f'
        self.assertEqual(expected, repr(fallback))

    def test_two_calls_default_repr(self):
        fallback = Fallback([f, g])
        expected = 'Fallback(Exception, callback=unit):\n[ 0] f\n[ 1] g'
        self.assertEqual(expected, repr(fallback))

    def test_two_calls_repr(self):
        fallback = Fallback([f, g], TypeError, callback=cb)
        expected = 'Fallback(TypeError, callback=cb):\n[ 0] f\n[ 1] g'
        self.assertEqual(expected, repr(fallback))


if __name__ == '__main__':
    unittest.main()

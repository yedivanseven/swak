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
        self.one = Fallback(f, TypeError, ValueError, callback=cb)

    def test_errors(self):
        self.assertSetEqual({TypeError, ValueError}, set(self.one.errors))

    def test_errors_deduplicated(self):
        fallback = Fallback(f, TypeError, TypeError, callback=cb)
        self.assertTupleEqual((TypeError, ), fallback.errors)

    def test_callback(self):
        self.assertIs(self.one.callback, cb)


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


# ToDo: Continue here!
class TestMagic(unittest.TestCase):
    pass


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

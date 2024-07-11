import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Route
from swak.funcflow.exceptions import RouteError
from swak.magic import ArgRepr, IndentRepr


def f(x):
    return x + 1


def g(x, y):
    return x, y


def h():
    return 3


def j(*x):
    return ()


def e():
    return ()


def r(*x):
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

    def __init__(self, *xs):
        super().__init__(*xs)

    def __call__(self, *_):
        raise AttributeError('Test!')


class TestAttributes(unittest.TestCase):

    def test_empty(self):
        route = Route()
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual((), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(0, route.n_args)

    def test_routes_empty(self):
        route = Route([])
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual((), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(0, route.n_args)

    def test_empty_route(self):
        route = Route([()], f)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((),), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f,), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(0, route.n_args)

    def test_empty_routes(self):
        route = Route([(), ()], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((), ()), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(0, route.n_args)

    def test_raises_on_route_empty_but_calls(self):
        expected = ('Number of callables (=1) must '
                    'match the number of routes (=0)!')
        with self.assertRaises(RouteError) as error:
            _ = Route([], f)
        self.assertEqual(expected, str(error.exception))

    def test_raises_on_route_but_no_calls(self):
        expected = ('Number of callables (=0) must '
                    'match the number of routes (=2)!')
        with self.assertRaises(RouteError) as error:
            _ = Route([1, 2], )
        self.assertEqual(expected, str(error.exception))

    def test_raises_on_route_calls_mismatch(self):
        expected = ('Number of callables (=3) must '
                    'match the number of routes (=2)!')
        with self.assertRaises(RouteError) as error:
            _ = Route([1, 2], f, f, f)
        self.assertEqual(expected, str(error.exception))

    def test_int_routes(self):
        route = Route([0, 3], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((0,), (3,)), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(4, route.n_args)

    def test_int_empty_routes(self):
        route = Route([2, ()], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((2,), ()), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(3, route.n_args)

    def test_tuple_routes(self):
        route = Route([(1, 3), (2, 0)], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((1, 3), (2, 0)), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(4, route.n_args)

    def test_tuple_empty_routes(self):
        route = Route([(), (2, 0)], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((), (2, 0)), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(3, route.n_args)

    def test_mixed_routes(self):
        route = Route([3, (2, 0)], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((3,), (2, 0)), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(4, route.n_args)

    def test_mixed_empty_routes(self):
        route = Route([3, (2, 0), ()], f, g, f)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((3,), (2, 0), ()), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g, f), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(4, route.n_args)

    def test_cast_routes(self):
        route = Route(['3', (2.3, 0)], f, g)
        self.assertTrue(hasattr(route, 'routes'))
        self.assertTupleEqual(((3,), (2, 0)), route.routes)
        self.assertTrue(hasattr(route, 'calls'))
        self.assertTupleEqual((f, g), route.calls)
        self.assertTrue(hasattr(route, 'n_args'))
        self.assertIsInstance(route.n_args, int)
        self.assertEqual(4, route.n_args)

    def test_raises_on_uncastable_route(self):
        expected = 'Routes must be integers or tuples thereof, not 3.1'
        with self.assertRaises(RouteError) as error:
            _ = Route(['3.1'], f)
        self.assertEqual(expected, str(error.exception))

    def test_raises_on_uncastable_routes(self):
        with self.assertRaises(RouteError):
            _ = Route([('foo', 'bar')], g)

    def test_raises_on_routes_wrong_type(self):
        with self.assertRaises(RouteError):
            _ = Route(f, g)


class TestUsage(unittest.TestCase):

    def test_callable(self):
        route = Route([3, (2, 0)], f, g)
        self.assertTrue(callable(route))

    def test_empty(self):
        route = Route()
        actual = route()
        self.assertTupleEqual((), actual)

    def test_empty_args(self):
        route = Route()
        actual = route(1, 2, 3)
        self.assertTupleEqual((), actual)

    def test_raises_on_too_few_args(self):
        expected = 'Number of arguments must be at least 4, not 3!'
        route = Route([3, (2, 0)], f, g)
        with self.assertRaises(RouteError) as error:
            _ = route(1, 2, 3)
        self.assertEqual(expected, str(error.exception))

    def test_too_many_arguments_ok(self):
        route = Route([3, (2, 0)], f, g)
        _ = route(1, 2, 3, 4, 5, 6)

    def test_empty_routes(self):
        route = Route([])
        actual = route()
        self.assertTupleEqual((), actual)

    def test_empty_routes_args(self):
        route = Route([])
        actual = route(1, 2, 3)
        self.assertTupleEqual((), actual)

    def test_empty_route(self):
        mock = Mock()
        route = Route([()], mock)
        _ = route()
        mock.assert_called_once()
        mock.assert_called_once_with()

    def test_empty_route_args(self):
        mock = Mock()
        route = Route([()], mock)
        _ = route(1, 2, 3)
        mock.assert_called_once()
        mock.assert_called_once_with()

    def test_multiple_empty_routes_args(self):
        mock1 = Mock()
        mock2 = Mock()
        mock3 = Mock()
        route = Route([(), (), ()], mock1, mock2, mock3)
        _ = route(1, 2, 3)
        mock1.assert_called_once()
        mock1.assert_called_once_with()
        mock2.assert_called_once()
        mock2.assert_called_once_with()
        mock3.assert_called_once()
        mock3.assert_called_once_with()

    def test_routing(self):
        mock1 = Mock()
        mock2 = Mock()
        mock3 = Mock()
        route = Route([1, (2, 0), (1, 3)], mock1, mock2, mock3)
        _ = route(1, 2, 3, 4, 5, 6)
        mock1.assert_called_once()
        mock1.assert_called_once_with(2)
        mock2.assert_called_once()
        mock2.assert_called_once_with(3, 1)
        mock3.assert_called_once()
        mock3.assert_called_once_with(2, 4)

    def test_return_object(self):
        route = Route([1], f)
        actual = route(1, 2, 4)
        self.assertIsInstance(actual, int)
        self.assertEqual(3, actual)

    def test_return_tuple(self):
        route = Route([(2, 0)], g)
        actual = route(1, 2, 4)
        self.assertTupleEqual((4, 1), actual)

    def test_return_objects(self):
        route = Route([2, 0], f, f)
        actual = route(1, 2, 4)
        self.assertTupleEqual((5, 2), actual)

    def test_return_tuples(self):
        route = Route([(2, 0), (0, 1)], g, g)
        actual = route(1, 2, 4)
        self.assertTupleEqual((4, 1, 1, 2), actual)

    def test_return_object_tuple(self):
        route = Route([1, (2, 0)], f, g)
        actual = route(1, 2, 4)
        self.assertTupleEqual((3, 4, 1), actual)

    def test_return_tuple_object(self):
        route = Route([(2, 0), 1], g, f)
        actual = route(1, 2, 4)
        self.assertTupleEqual((4, 1, 3), actual)

    def test_call_empty_return(self):
        route = Route([()], h)
        actual = route(1, 2, 3)
        self.assertEqual(3, actual)

    def test_call_no_return(self):
        route = Route([(1, 2)], j)
        actual = route(1, 2, 3)
        self.assertTupleEqual((), actual)

    def test_empty_call_empty_return(self):
        route = Route([()], e)
        actual = route(1, 2, 3)
        self.assertTupleEqual((), actual)

    def test_return_object_and_empty(self):
        route = Route([0, ()], f, e)
        actual = route(1)
        self.assertIsInstance(actual, int)
        self.assertEqual(2, actual)

    def test_return_empty_and_object(self):
        route = Route([(), 0], e, f)
        actual = route(1)
        self.assertIsInstance(actual, int)
        self.assertEqual(2, actual)

    def test_return_tuple_and_empty(self):
        route = Route([0, 0, ()], f, f, e)
        actual = route(1)
        self.assertTupleEqual((2, 2), actual)

    def test_return_empty_and_tuple(self):
        route = Route([(), 0, 0], e, f, f)
        actual = route(1)
        self.assertTupleEqual((2, 2), actual)

    def test_raises(self):
        route = Route([3, (2, 0)], f, r)
        expected = ("Error executing\n"
                    "r\n"
                    "in route #1 (2, 0) of\n"
                    "Route:\n"
                    "[ 0] f\n"
                    "[ 1] r\n"
                    "AttributeError:\n"
                    "Test!")
        with self.assertRaises(RouteError) as error:
            _ = _ = route(1, 2, 3, 4)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_argrepr(self):
        route = Route([3, (2, 0)], f, A(1))
        expected = ("Error executing\n"
                    "A(1)\n"
                    "in route #1 (2, 0) of\n"
                    "Route:\n"
                    "[ 0] f\n"
                    "[ 1] A(1)\n"
                    "AttributeError:\n"
                    "Test!")
        with self.assertRaises(RouteError) as error:
            _ = _ = route(1, 2, 3, 4)
        self.assertEqual(expected, str(error.exception))

    def test_error_msg_indentrepr(self):
        route = Route([3, (2, 0)], f, Ind(1))
        expected = ("Error executing\n"
                    "Ind:\n"
                    "[ 0] 1\n"
                    "in route #1 (2, 0) of\n"
                    "Route:\n"
                    "[ 0] f\n"
                    "[ 1] Ind:\n"
                    "     [ 0] 1\n"
                    "AttributeError:\n"
                    "Test!")
        with self.assertRaises(RouteError) as error:
            _ = _ = route(1, 2, 3, 4)
        self.assertEqual(expected, str(error.exception))


class TestMagic(unittest.TestCase):

    def setUp(self):
        self.calls = f, g, lambda *x: x, Cls.m, A(1), Call()
        self.routes = (1,), (2,), (3,), (4,), (5,), (6,)
        self.route = Route(self.routes, *self.calls)

    def test_iter(self):
        for i, call in enumerate(self.route):
            self.assertIs(self.calls[i], call)

    def test_len(self):
        self.assertEqual(6, len(self.route))

    def test_bool_empty(self):
        self.assertFalse(Route())

    def test_bool_non_empty(self):
        self.assertTrue(self.route)

    def test_contains_true(self):
        self.assertIn(g, self.route)

    def test_contains_false(self):
        self.assertNotIn(Ind(1, 2, 3), self.route)

    def test_reversed_type(self):
        self.assertIsInstance(reversed(self.route), Route)

    def test_reversed_routes(self):
        expected = tuple(reversed(self.routes))
        self.assertTupleEqual(expected, reversed(self.route).routes)

    def test_reversed_calls(self):
        expected = tuple(reversed(self.calls))
        self.assertTupleEqual(expected, reversed(self.route).calls)

    def test_getitem_int(self):
        for i, call in enumerate(self.calls):
            self.assertIs(call, self.route[i])

    def test_getitem_single_slice(self):
        for i, _ in enumerate(zip(self.routes, self.calls)):
            self.assertIsInstance(self.route[i:i + 1], Route)
            self.assertTupleEqual(
                self.routes[i:i + 1],
                self.route[i:i + 1].routes
            )
            self.assertTupleEqual(
                self.calls[i:i + 1],
                self.route[i:i + 1].calls
            )

    def test_getitem_multiple_slice(self):
        self.assertIsInstance(self.route[1:4], Route)
        self.assertTupleEqual(self.routes[1:4], self.route[1:4].routes)
        self.assertTupleEqual(self.calls[1:4], self.route[1:4].calls)

    def test_equal_self(self):
        self.assertEqual(self.route, self.route)

    def test_equal_same_content(self):
        route = Route(self.routes, *self.calls)
        self.assertEqual(self.route, route)

    def test_equal_false_different_routes(self):
        routes = ((4, 5),) + self.routes[:-1]
        route = Route(routes, *self.calls)
        self.assertFalse(self.route == route)

    def test_equal_false_different_calls(self):
        calls = (Ind(1, 2, 3),) + self.calls[:-1]
        route = Route(self.routes, *calls)
        self.assertFalse(self.route == route)

    def test_equal_false_different_type(self):
        self.assertFalse(self.route == 1)

    def test_not_equal_different_routes(self):
        routes = ((4, 5),) + self.routes[:-1]
        route = Route(routes, *self.calls)
        self.assertNotEqual(self.route, route)

    def test_not_equal_different_calls(self):
        calls = (Ind(1, 2, 3),) + self.calls[:-1]
        route = Route(self.routes, *calls)
        self.assertNotEqual(self.route, route)

    def test_not_equal_different_type(self):
        self.assertNotEqual(self.route, 1)

    def test_not_equal_false_self(self):
        self.assertFalse(self.route != self.route)

    def test_not_equal_false_same_content(self):
        route = Route(self.routes, *self.calls)
        self.assertFalse(self.route != route)

    def test_add_empty(self):
        route = self.route + Route()
        self.assertIsInstance(route, Route)
        self.assertEqual(self.route, route)

    def test_add(self):
        route = self.route + Route([(1, 2), 3], f, g)
        self.assertIsInstance(route, Route)
        self.assertTupleEqual(self.routes + ((1, 2), (3,)), route.routes)
        self.assertTupleEqual(self.calls + (f, g), route.calls)


class TestMisc(unittest.TestCase):

    def test_type_annotation(self):
        _ = Route[[int, float, bool, str], dict](['3', (2.3, 0)], f, g)

    def test_type_annotation_tuple(self):
        _ = Route[[int, float], tuple[dict, str]]([2, (1, 0)], f, g)

    def test_pickle_works(self):
        route = Route([3, (2, 0)], f, g)
        _ = pickle.dumps(route)

    def test_pickle_raises_with_lambdas(self):
        route = Route([3, (2, 0)], lambda x: x + 1, lambda x: x**2)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(route)

    def test_flat(self):
        route = Route(
            [0, 1, 2, 3, 4, 5, 6, 7],
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
            "Route:\n"
            "[ 0] lambda\n"
            "[ 1] f\n"
            "[ 2] Cls\n"
            "[ 3] Cls.c\n"
            "[ 4] Cls.m\n"
            "[ 5] Cls.s\n"
            "[ 6] Call(...)\n"
            "[ 7] A('foo')"
        )
        self.assertEqual(expected, repr(route))

    def test_nested(self):
        route = Route(
            [0, 1, 2, 3, 4, 5, 6, 7],
            lambda x: x,
            f,
            Cls,
            Cls.c,
            Cls().m,
            Cls().s,
            Call(),
            A('foo')
        )
        outer = Route([0, 1], route, route)
        expected = (
            "Route:\n"
            "[ 0] Route:\n"
            "     [ 0] lambda\n"
            "     [ 1] f\n"
            "     [ 2] Cls\n"
            "     [ 3] Cls.c\n"
            "     [ 4] Cls.m\n"
            "     [ 5] Cls.s\n"
            "     [ 6] Call(...)\n"
            "     [ 7] A('foo')\n"
            "[ 1] Route:\n"
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

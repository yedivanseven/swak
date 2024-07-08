import unittest
import pickle
from unittest.mock import Mock
from swak.funcflow import Map
from swak.funcflow.exceptions import MapError


def plus_2(x: int) -> int:
    return x + 2


def plus(x: int, y: int) -> int:
    return x + y


class TestDefaultAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Map(plus_2)

    def test_has_transform(self):
        m = Map(plus_2)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = Map(plus_2)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = Map(plus_2)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_is_none(self):
        m = Map(plus_2)
        self.assertIsNone(m.wrapper)

    def test_callable(self):
        m = Map(plus_2)
        self.assertTrue(callable(m))


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = Map(plus_2)
        self.m2 = Map(plus)

    def test_empty_list(self):
        actual = self.m1([])
        self.assertListEqual([], actual)

    def test_empty_lists(self):
        actual = self.m1([], [])
        self.assertListEqual([], actual)

    def test_empty_tuple(self):
        actual = self.m1(tuple())
        self.assertTupleEqual(tuple(), actual)

    def test_empty_tuples(self):
        actual = self.m1(tuple(), tuple())
        self.assertTupleEqual(tuple(), actual)

    def test_empty_set(self):
        actual = self.m1(set())
        self.assertSetEqual(set(), actual)

    def test_empty_sets(self):
        actual = self.m1(set(), set())
        self.assertSetEqual(set(), actual)

    def test_empty_mixed(self):
        actual = self.m1([], tuple(), set())
        self.assertListEqual([], actual)

    def test_call_called(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1])
        mock.assert_called_once()

    def test_transform_called_correctly_one_iterable(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1])
        mock.assert_called_once_with(1)

    def test_transform_called_correctly_two_iterables(self):
        mock = Mock()
        m = Map(mock)
        _ = m([1], (2,))
        mock.assert_called_once_with(1, 2)

    def test_transform_called_correctly_three_iterables(self):
        mock = Mock()
        m = Map(mock)
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
        m = Map(lambda x: 1 / x)
        with self.assertRaises(MapError):
            _ = m([1, 0, 2])

    def test_wrong_iterable_raises(self):
        with self.assertRaises(MapError):
            _ = self.m1(iter([1, 2, 3]))


class TestWrapperAttributes(unittest.TestCase):

    def test_instantiation(self):
        _ = Map(plus_2, tuple)

    def test_has_transform(self):
        m = Map(plus_2, tuple)
        self.assertTrue(hasattr(m, 'transform'))

    def test_transform_correct(self):
        m = Map(plus_2, tuple)
        self.assertIs(m.transform, plus_2)

    def test_has_wrapper(self):
        m = Map(plus_2, tuple)
        self.assertTrue(hasattr(m, 'wrapper'))

    def test_wrapper_correct(self):
        m = Map(plus_2, tuple)
        self.assertIs(m.wrapper, tuple)

    def test_callable(self):
        m = Map(plus_2, tuple)
        self.assertTrue(callable(m))


class TestWrapperUsage(unittest.TestCase):

    def setUp(self):
        self.m1 = Map(plus_2, list)
        self.m2 = Map(plus, list)

    def test_wrapper_called(self):
        mock = Mock()
        m = Map(plus_2, mock)
        _ = m([1, 2, 3])
        mock.assert_called_once()

    def test_wrapper_called_correctly(self):
        mock = Mock()
        m = Map(plus_2, mock)
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
        m = Map(plus_2, int)
        with self.assertRaises(MapError):
            _ = m([1, 2, 3])


class TestMisc(unittest.TestCase):

    def test_default_pickle_works(self):
        m = Map(plus_2)
        _ = pickle.dumps(m)

    def test_default_pickle_raises_lambda(self):
        m = Map(lambda x: x + 2)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_wrapper_pickle_works(self):
        m = Map(plus_2, tuple)
        _ = pickle.dumps(m)

    def test_wrapper_pickle_raises_lambda(self):
        m = Map(plus_2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_pickle_raises_lambda(self):
        m = Map(lambda x: x + 2, lambda x: tuple(x))
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(m)

    def test_default_repr(self):
        m = Map(plus_2)
        self.assertEqual('Map(plus_2, None)', repr(m))

    def test_wrapper_repr(self):
        m = Map(plus_2, tuple)
        self.assertEqual('Map(plus_2, tuple)', repr(m))

    def test_annotated(self):
        _ = Map[[int, bool], float, list]

    def test_annotated_return_elements(self):
        _ = Map[[int, bool], float, list[float]]


if __name__ == '__main__':
    unittest.main()

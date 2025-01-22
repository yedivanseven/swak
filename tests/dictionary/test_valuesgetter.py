import unittest
import pickle
from unittest.mock import Mock
from swak.dictionary import ValuesGetter


class TestDefaultAttributes(unittest.TestCase):

    def test_empty(self):
        values_from = ValuesGetter()
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_empty_list(self):
        values_from = ValuesGetter([])
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_one_int_key(self):
        values_from = ValuesGetter(1)
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((1,), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_one_str_key(self):
        values_from = ValuesGetter('one')
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one',), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_three_int_keys(self):
        values_from = ValuesGetter(1, 2, 3)
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((1, 2, 3), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_one_int_key_two_int_keys(self):
        values_from = ValuesGetter([1], 2, 3)
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((1, 2, 3), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_two_int_keys_one_int_key(self):
        values_from = ValuesGetter([1, 2], 3)
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((1, 2, 3), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_three_int_keys(self):
        values_from = ValuesGetter([1, 2, 3])
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual((1, 2, 3), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_three_str_keys(self):
        values_from = ValuesGetter('one', 'two', 'three')
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one', 'two', 'three'), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_one_str_key_two_str_keys(self):
        values_from = ValuesGetter(['one'], 'two', 'three')
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one', 'two', 'three'), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_two_str_keys_one_str_key(self):
        values_from = ValuesGetter(['one', 'two'], 'three')
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one', 'two', 'three'), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_list_three_str_keys(self):
        values_from = ValuesGetter(['one', 'two', 'three'])
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one', 'two', 'three'), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_mixed_keys(self):
        values_from = ValuesGetter(['one', 2], 'three', 4)
        self.assertTrue(hasattr(values_from, 'keys'))
        self.assertTupleEqual(('one', 2, 'three', 4), values_from.keys)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertTrue(callable(values_from.wrapper))

    def test_non_hashables_raise(self):
        with self.assertRaises(TypeError):
            _ = ValuesGetter([[], []])

    def test_non_hashable_args_raise(self):
        with self.assertRaises(TypeError):
            _ = ValuesGetter(1, [])


class TestDefaultUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_callable(self):
        values_from = ValuesGetter()
        self.assertTrue(callable(values_from))

    def test_empty(self):
        values_from = ValuesGetter()
        actual = values_from(self.d)
        self.assertListEqual([], actual)

    def test_single(self):
        values_from = ValuesGetter(3)
        actual = values_from(self.d)
        self.assertListEqual([self.d[3]], actual)

    def test_ordered(self):
        values_from = ValuesGetter(2, 3)
        actual = values_from(self.d)
        self.assertListEqual([self.d[2], self.d[3]], actual)

    def test_reversed(self):
        values_from = ValuesGetter(3, 2)
        actual = values_from(self.d)
        self.assertListEqual([self.d[3], self.d[2]], actual)

    def test_duplicates_only(self):
        values_from = ValuesGetter(4, 4)
        actual = values_from(self.d)
        self.assertListEqual([self.d[4], self.d[4]], actual)

    def test_duplicates_and_single(self):
        values_from = ValuesGetter(2, 4, 4, 1)
        actual = values_from(self.d)
        expected = [self.d[2], self.d[4], self.d[4], self.d[1]]
        self.assertListEqual(expected, actual)


class TestWrapperAttribute(unittest.TestCase):

    def test_wrapper_correct(self):
        values_from = ValuesGetter(2, 3, wrapper=set)
        self.assertTrue(hasattr(values_from, 'wrapper'))
        self.assertIs(values_from.wrapper, set)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_wrapper_called_once(self):
        mock = Mock()
        values_from = ValuesGetter[mock](2, 3, wrapper=mock)
        _ = values_from(self.d)
        mock.assert_called_once()
        mock.assert_called_once_with((self.d[2], self.d[3]))

    def test_wrapper_called_correctly(self):
        mock = Mock()
        values_from = ValuesGetter[mock](2, 3, wrapper=mock)
        _ = values_from(self.d)
        mock.assert_called_once_with((self.d[2], self.d[3]))

    def test_wrapper_wraps_example(self):
        values_from = ValuesGetter[set](2, 3, wrapper=set)
        actual = values_from(self.d)
        self.assertSetEqual({self.d[2], self.d[3]}, actual)


class TestMagic(unittest.TestCase):

    def setUp(self):
        self.keys = 1, 2, 3
        self.empty = ValuesGetter()
        self.values_from = ValuesGetter(*self.keys)

    def test_len(self):
        self.assertEqual(0, len(self.empty))
        self.assertEqual(len(self.keys), len(self.values_from))

    def test_bool(self):
        self.assertFalse(self.empty)
        self.assertTrue(self.values_from)

    def test_reversed(self):
        expected = tuple(reversed(self.keys))
        actual = reversed(self.values_from)
        self.assertIsInstance(actual, ValuesGetter)
        self.assertTupleEqual(expected, actual.keys)

    def test_iter(self):
        for i, g in enumerate(self.values_from):
            self.assertIsInstance(g, ValuesGetter)
            self.assertEqual(1, len(g))
            self.assertEqual(self.keys[i], g.keys[0])

    def test_contains(self):
        self.assertTrue(1 in self.values_from)
        self.assertFalse(4 in self.values_from)

    def test_getitem_int(self):
        g = self.values_from[0]
        self.assertIsInstance(g, ValuesGetter)
        self.assertEqual(1, len(g))
        self.assertEqual(self.keys[0], g.keys[0])

    def test_getitem_slice(self):
        g = self.values_from[:2]
        self.assertIsInstance(g, ValuesGetter)
        self.assertEqual(2, len(g))
        self.assertTupleEqual(self.keys[:2], g.keys[:2])

    def test_equality_true_other(self):
        self.assertEqual(self.values_from, ValuesGetter(*self.keys))

    def test_equality_true_self(self):
        self.assertEqual(self.values_from, ValuesGetter(*self.keys))

    def test_equality_false_wrong_type(self):
        self.assertFalse(self.values_from == 4)

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.values_from == ValuesGetter(4, 5))

    def test_inequality_false_other(self):
        self.assertFalse(self.values_from != ValuesGetter(*self.keys))

    def test_inequality_false_self(self):
        self.assertFalse(self.values_from != self.values_from)

    def test_inequality_true_wrong_type(self):
        self.assertNotEqual(self.values_from, 4)

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.values_from, ValuesGetter(4, 5))

    def test_add_key(self):
        getter = self.values_from + 4
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4), getter.keys)

    def test_add_empty_keys(self):
        getter = self.values_from + []
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_add_keys(self):
        getter = self.values_from + [4, 5]
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4, 5), getter.keys)

    def test_add_empty_self(self):
        getter = self.values_from + ValuesGetter()
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_add_self(self):
        getter = self.values_from + ValuesGetter(4, 5)
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4, 5), getter.keys)

    def test_add_unhashable_raises(self):
        with self.assertRaises(TypeError):
            _ = self.values_from + [[4], [5]]

    def test_radd_key(self):
        getter = 4 + self.values_from
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((4, *self.keys), getter.keys)

    def test_radd_empty_keys(self):
        getter = [] + self.values_from
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_radd_keys(self):
        getter = [4, 5] + self.values_from
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((4, 5, *self.keys), getter.keys)

    def test_radd_unhashable_raises(self):
        with self.assertRaises(TypeError):
            _ = [[4], [5]] + self.values_from


class TestMisc(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_pickle_works(self):
        getter = ValuesGetter(2, 3)
        _ = pickle.loads(pickle.dumps(getter))

    def test_pickle_raised_lambda(self):
        values_from = ValuesGetter(2, 3, wrapper=lambda x: list(x))
        actual = values_from(self.d)
        self.assertListEqual([self.d[2], self.d[3]], actual)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(values_from))

    def test_default_representation(self):
        get = ValuesGetter(1, '2', 3, '4', 5)
        expected = "ValuesGetter(1, '2', 3, '4', 5, wrapper=list)"
        self.assertEqual(expected, repr(get))

    def test_wrapper_representation(self):
        get = ValuesGetter(1, '2', 3, '4', 5, wrapper=set)
        expected = "ValuesGetter(1, '2', 3, '4', 5, wrapper=set)"
        self.assertEqual(expected, repr(get))

    def test_type_annotation(self):
        _ = ValuesGetter[list](2, 3)

    def test_wrapper_type_annotation(self):
        _ = ValuesGetter[set](2, 3, wrapper=set)


if __name__ == '__main__':
    unittest.main()

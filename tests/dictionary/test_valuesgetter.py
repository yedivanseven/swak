import unittest
import pickle
from swak.dictionary import ValuesGetter


class TestBasicAttributes(unittest.TestCase):

    def setUp(self) -> None:
        self.getter = ValuesGetter(1, 2, 3)

    def test_has_keys(self):
        self.assertTrue(hasattr(self.getter, 'keys'))

    def test_keys_correct(self):
        self.assertTupleEqual((1, 2, 3), self.getter.keys)

    def test_has_wrapper(self):
        self.assertTrue(hasattr(self.getter, 'wrapper'))

    def test_wrapper_callable(self):
        self.assertTrue(callable(self.getter.wrapper))

    def test_wrap_callable_with_tuple(self):
        _ = self.getter.wrapper((1, 2))


class TestBasicUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

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

    def test_representation(self):
        get = ValuesGetter(1, '2', 3, '4', 5)
        expected = "ValuesGetter(1, '2', 3, '4', 5, wrapper=list)"
        self.assertEqual(expected, repr(get))


class TestWrapperAttribute(unittest.TestCase):

    def test_wrapper_default(self):
        getter = ValuesGetter(2, 3)
        self.assertIs(getter.wrapper, list)

    def test_wrapper_correct(self):
        getter = ValuesGetter(2, 3, wrapper=set)
        self.assertIs(getter.wrapper, set)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_takes_wrap_kwarg(self):
        _ = ValuesGetter(2, 3, wrapper=set)

    def test_takes_generic_default(self):
        _ = ValuesGetter[list](2, 3)

    def test_wrapper_wraps(self):
        values_from = ValuesGetter[set](2, 3, wrapper=set)
        actual = values_from(self.d)
        self.assertSetEqual({self.d[2], self.d[3]}, actual)


class TestMisc(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_pickle_works(self):
        getter = ValuesGetter(2, 3)
        _ = pickle.dumps(getter)

    def test_pickle_raised_lambda(self):
        values_from = ValuesGetter(2, 3, wrapper=lambda x: list(x))
        actual = values_from(self.d)
        self.assertListEqual([self.d[2], self.d[3]], actual)
        with self.assertRaises(AttributeError):
            _ = pickle.dumps(values_from)


class TestMagic(unittest.TestCase):

    def setUp(self):
        self.keys = 1, 2, 3
        self.empty = ValuesGetter()
        self.getter = ValuesGetter(*self.keys)

    def test_len(self):
        self.assertEqual(0, len(self.empty))
        self.assertEqual(len(self.keys), len(self.getter))

    def test_bool(self):
        self.assertFalse(self.empty)
        self.assertTrue(self.getter)

    def test_reversed(self):
        expected = tuple(reversed(self.keys))
        actual = reversed(self.getter)
        self.assertIsInstance(actual, ValuesGetter)
        self.assertTupleEqual(expected, actual.keys)

    def test_iter(self):
        for i, g in enumerate(self.getter):
            self.assertIsInstance(g, ValuesGetter)
            self.assertEqual(1, len(g))
            self.assertEqual(self.keys[i], g.keys[0])

    def test_contains(self):
        self.assertTrue(1 in self.getter)
        self.assertFalse(4 in self.getter)

    def test_getitem_int(self):
        g = self.getter[0]
        self.assertIsInstance(g, ValuesGetter)
        self.assertEqual(1, len(g))
        self.assertEqual(self.keys[0], g.keys[0])

    def test_getitem_slice(self):
        g = self.getter[:2]
        self.assertIsInstance(g, ValuesGetter)
        self.assertEqual(2, len(g))
        self.assertTupleEqual(self.keys[:2], g.keys[:2])

    def test_equality_true_other(self):
        self.assertEqual(self.getter, ValuesGetter(*self.keys))

    def test_equality_true_self(self):
        self.assertEqual(self.getter, ValuesGetter(*self.keys))

    def test_equality_false_wrong_type(self):
        self.assertFalse(self.getter == 4)

    def test_equality_false_wrong_content(self):
        self.assertFalse(self.getter == ValuesGetter(4, 5))

    def test_inequality_false_other(self):
        self.assertFalse(self.getter != ValuesGetter(*self.keys))

    def test_inequality_false_self(self):
        self.assertFalse(self.getter != self.getter)

    def test_inequality_true_wrong_type(self):
        self.assertNotEqual(self.getter, 4)

    def test_inequality_true_wrong_content(self):
        self.assertNotEqual(self.getter, ValuesGetter(4, 5))

    def test_add_key(self):
        getter = self.getter + 4
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4), getter.keys)

    def test_add_empty_keys(self):
        getter = self.getter + []
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_add_keys(self):
        getter = self.getter + [4, 5]
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4, 5), getter.keys)

    def test_add_empty_self(self):
        getter = self.getter + ValuesGetter()
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_add_self(self):
        getter = self.getter + ValuesGetter(4, 5)
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((*self.keys, 4, 5), getter.keys)

    def test_radd_key(self):
        getter = 4 + self.getter
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((4, *self.keys), getter.keys)

    def test_radd_empty_keys(self):
        getter = [] + self.getter
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual(self.keys, getter.keys)

    def test_radd_keys(self):
        getter = [4, 5] + self.getter
        self.assertIsInstance(getter, ValuesGetter)
        self.assertTupleEqual((4, 5, *self.keys), getter.keys)


if __name__ == '__main__':
    unittest.main()

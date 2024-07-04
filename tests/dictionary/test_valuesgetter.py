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

    def test_has_n_items(self):
        self.assertTrue(hasattr(self.getter, 'n_items'))

    def test_n_items_correct_type(self):
        self.assertIsInstance(self.getter.n_items, int)

    def test_n_items_correct_value(self):
        self.assertEqual(3, self.getter.n_items)


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


if __name__ == '__main__':
    unittest.main()

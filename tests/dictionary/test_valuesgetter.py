import unittest
from swak.dictionary import ValuesGetter


class TestBasicAttributes(unittest.TestCase):

    def setUp(self) -> None:
        self.getter = ValuesGetter(1, 2, 3)

    def test_has_keys(self):
        self.assertTrue(hasattr(self.getter, 'keys'))

    def test_keys_correct(self):
        self.assertTupleEqual((1, 2, 3), self.getter.keys)

    def test_has_wrap(self):
        self.assertTrue(hasattr(self.getter, 'wrap'))

    def test_wrap_callable(self):
        self.assertTrue(callable(self.getter.wrap))

    def test_wrap_callable_with_tuple(self):
        _ = self.getter.wrap((1, 2))

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
        values = values_from(self.d)
        self.assertListEqual([], values)

    def test_single(self):
        values_from = ValuesGetter(3)
        values = values_from(self.d)
        self.assertListEqual([self.d[3]], values)

    def test_ordered(self):
        values_from = ValuesGetter(2, 3)
        values = values_from(self.d)
        self.assertListEqual([self.d[2], self.d[3]], values)

    def test_reversed(self):
        values_from = ValuesGetter(3, 2)
        values = values_from(self.d)
        self.assertListEqual([self.d[3], self.d[2]], values)

    def test_duplicates_only(self):
        values_from = ValuesGetter(4, 4)
        values = values_from(self.d)
        self.assertListEqual([self.d[4], self.d[4]], values)

    def test_duplicates_and_single(self):
        values_from = ValuesGetter(2, 4, 4, 1)
        values = values_from(self.d)
        should = [self.d[2], self.d[4], self.d[4], self.d[1]]
        self.assertListEqual(should, values)

    def test_representation(self):
        get = ValuesGetter(1, '2', 3, '4', 5)
        expected = "ValuesGetter(1, '2', 3, '4', 5, wrap=list)"
        self.assertEqual(expected, repr(get))


class TestWrapperAttribute(unittest.TestCase):

    def test_wrapper_default(self):
        getter = ValuesGetter(2, 3)
        self.assertIs(getter.wrap, list)

    def test_wrapper_correct(self):
        getter = ValuesGetter(2, 3, wrap=set)
        self.assertIs(getter.wrap, set)


class TestWrapperUsage(unittest.TestCase):

    def setUp(self) -> None:
        self.d = {1: 'hello', 2: 'world', 3: 'foo', 4: 'bar'}

    def test_takes_wrap_kwarg(self):
        _ = ValuesGetter(2, 3, wrap=set)

    def test_takes_generic_default(self):
        _ = ValuesGetter[list](2, 3)

    def test_wrapper_wraps(self):
        values_from = ValuesGetter[set](2, 3, wrap=set)
        values = values_from(self.d)
        self.assertSetEqual({self.d[2], self.d[3]}, values)


if __name__ == '__main__':
    unittest.main()

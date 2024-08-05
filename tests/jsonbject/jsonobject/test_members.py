import unittest
import pandas as pd
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Custom:

    def __init__(self, *_):
        pass

    def __repr__(self):
        return 'custom'


class Simple(JsonObject):
    a: int = 1
    b: str = 'foo'
    c: Custom = True


class Option(JsonObject):
    a: Maybe[int](int) = None


class TestMembers(unittest.TestCase):

    def setUp(self):
        self.simple = Simple()

    def test_has_as_json(self):
        self.assertTrue(hasattr(self.simple, 'as_json'))

    def test_as_json_type(self):
        self.assertIsInstance(self.simple.as_json, dict)

    def test_as_json_value(self):
        expected = {'a': 1, 'b': 'foo', 'c': 'custom'}
        self.assertDictEqual(expected, self.simple.as_json)

    def test_has_as_dtype(self):
        self.assertTrue(hasattr(self.simple, 'as_dtype'))

    def test_as_dtype_type(self):
        self.assertIsInstance(self.simple.as_dtype, str)

    def test_as_dtype_value(self):
        expected = '{"a": 1, "b": "foo", "c": "custom"}'
        self.assertEqual(expected, self.simple.as_dtype)

    def test_has_as_series(self):
        self.assertTrue(hasattr(self.simple, 'as_series'))

    def test_as_series_type(self):
        self.assertIsInstance(self.simple.as_series, pd.Series)

    def test_as_series_value(self):
        expected = pd.Series(
            {'a': 1, 'b': 'foo', 'c': self.simple.c},
            name='Simple'
        )
        pd.testing.assert_series_equal(expected, self.simple.as_series)

    # Do we even need this method?
    def test_has_get(self):
        self.assertTrue(hasattr(self.simple, 'get'))

    def test_get_callable(self):
        self.assertTrue(callable(self.simple.get))

    def test_get_value(self):
        self.assertIsInstance(self.simple.get('a'), int)
        self.assertEqual(1, self.simple.get('a'))
        self.assertEqual('foo', self.simple.get('b'))
        self.assertIsInstance(self.simple.get('c'), Custom)

    def test_get_non_existent(self):
        default = self.simple.get('d')
        self.assertIsNone(default)

    def test_get_default(self):
        default = self.simple.get('d', 'default')
        self.assertIsInstance(default, str)
        self.assertEqual('default', default)

    def test_get_none_field(self):
        default = Option().get('a', 'default')
        self.assertIsNone(default)
    ###########################################################################

    def test_has_keys(self):
        self.assertTrue(hasattr(self.simple, 'keys'))

    def test_keys_callable(self):
        self.assertTrue(callable(self.simple.keys))

    def test_keys_value(self):
        self.assertListEqual(['a', 'b', 'c'], list(self.simple.keys()))

    def test_has_ignore_extra(self):
        self.assertTrue(hasattr(self.simple, '__ignore_extra__'))

    def test_ignore_extra_type(self):
        self.assertIsInstance(self.simple.__ignore_extra__, bool)

    def test_has_raise_extra(self):
        self.assertTrue(hasattr(self.simple, '__raise_extra__'))

    def test_raise_extra_type(self):
        self.assertIsInstance(self.simple.__raise_extra__, bool)

    def test_has_respect_none(self):
        self.assertTrue(hasattr(self.simple, '__respect_none__'))

    def test_respect_none_type(self):
        self.assertIsInstance(self.simple.__respect_none__, bool)


if __name__ == '__main__':
    unittest.main()

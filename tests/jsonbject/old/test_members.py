import unittest
import pandas as pd
from swak.jsonobject import JsonObject


class Simple(JsonObject):
    a: int = 1
    b: str = 'foo'


class Dummy:

    def __init__(self, *args):
        pass


class Custom(JsonObject):
    d: Dummy = 1


class TestMembers(unittest.TestCase):

    def test_as_json(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'as_json'))
        self.assertIsInstance(simple.as_json, dict)
        self.assertDictEqual({'a': 1, 'b': 'foo'}, simple.as_json)

    def test_as_json_custom(self):
        custom = Custom()
        self.assertTrue(hasattr(custom, 'as_json'))
        self.assertIsInstance(custom.as_json, dict)
        self.assertIsInstance(custom.as_json['d'], str)
        self.assertEqual(custom.as_json['d'], str(custom.d))

    def test_as_dtype(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'as_dtype'))
        self.assertIsInstance(simple.as_dtype, str)
        self.assertEqual('{"a": 1, "b": "foo"}', simple.as_dtype)

    def test_as_dtype_custom(self):
        custom = Custom()
        self.assertTrue(hasattr(custom, 'as_dtype'))
        self.assertIsInstance(custom.as_dtype, str)
        self.assertEqual(f'{{"d": "{str(custom.d)}"}}', custom.as_dtype)

    def test_as_series(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'as_series'))
        self.assertIsInstance(simple.as_series, pd.Series)
        expected = pd.Series({'a': 1, 'b': 'foo'}, name='Simple')
        pd.testing.assert_series_equal(expected, simple.as_series)

    def test_as_series_custom(self):
        custom = Custom()
        self.assertTrue(hasattr(custom, 'as_series'))
        self.assertIsInstance(custom.as_series, pd.Series)
        expected = pd.Series({'d': custom.d}, name='Custom')
        pd.testing.assert_series_equal(expected, custom.as_series)

    def test_get(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'get'))
        self.assertIsInstance(simple.get('a'), int)
        self.assertEqual(1, simple.get('a'))
        self.assertIsInstance(simple.get('b'), str)
        self.assertEqual('foo', simple.get('b'))

    def test_get_non_existent(self):
        simple = Simple()
        default = simple.get('c')
        self.assertIsNone(default)

    def test_get_default(self):
        simple = Simple()
        default = simple.get('c', 'default')
        self.assertIsInstance(default, str)
        self.assertEqual('default', default)

    def test_keys(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'keys'))
        self.assertListEqual(['a', 'b'], list(simple.keys()))

    def test_ignore_extra(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'ignore_extra'))
        self.assertIsInstance(simple.ignore_extra, bool)

    def test_raise_extra(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'raise_extra'))
        self.assertIsInstance(simple.ignore_extra, bool)

    def test_respect_none(self):
        simple = Simple()
        self.assertTrue(hasattr(simple, 'respect_none'))
        self.assertIsInstance(simple.respect_none, bool)


if __name__ == '__main__':
    unittest.main()

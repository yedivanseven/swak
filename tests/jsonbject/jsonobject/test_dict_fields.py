import unittest
from swak.jsonobject import JsonObject


class Flat(JsonObject):
    c: int = 42
    d: dict


class Nested(JsonObject):
    a: str = 'foo'
    b: Flat


class TestFlat(unittest.TestCase):

    def test_empty(self):
        flat = Flat({'d': {}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({}, flat['d'])

    def test_str_keys(self):
        flat = Flat({'d': {'foo': 1}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({'foo': 1}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({'foo': 1}, flat['d'])
        self.assertEqual(1, flat.d['foo'])
        self.assertEqual(1, flat['d']['foo'])
        self.assertEqual(1, flat['d.foo'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo

    def test_non_str_keys(self):
        flat = Flat({'d': {2: 3}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({2: 3}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({2: 3}, flat['d'])
        self.assertEqual(3, flat.d[2])
        self.assertEqual(3, flat['d'][2])
        with self.assertRaises(KeyError):
            _ = flat['d.2']


class TestNested(unittest.TestCase):

    def test_empty(self):
        nested = Nested({'b': {'d': {}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({}, nested['b']['d'])
        self.assertIsInstance(nested['b.d'], dict)
        self.assertDictEqual({}, nested['b.d'])

    def test_str_keys(self):
        nested = Nested({'b': {'d': {'foo': 1}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({'foo': 1}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({'foo': 1}, nested['b']['d'])
        self.assertEqual(1, nested.b.d['foo'])
        self.assertEqual(1, nested['b']['d']['foo'])
        self.assertEqual(1, nested['b.d.foo'])
        with self.assertRaises(AttributeError):
            _ = nested.b.d.foo

    def test_non_str_keys(self):
        nested = Nested({'b': {'d': {2: 3}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({2: 3}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({2: 3}, nested['b']['d'])
        self.assertIsInstance(nested['b.d'], dict)
        self.assertDictEqual({2: 3}, nested['b.d'])
        self.assertEqual(3, nested.b.d[2])
        with self.assertRaises(KeyError):
            _ = nested['b.d.2']


if __name__ == '__main__':
    unittest.main()

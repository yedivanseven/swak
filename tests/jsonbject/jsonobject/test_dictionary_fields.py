import unittest
from swak.jsonobject import JsonObject

# ToDo: Continue here!
class Flat(JsonObject):
    c: int
    d: dict


class Nested(JsonObject):
    a: str
    b: Flat


class TestFlat(unittest.TestCase):

    def test_str_keys(self):
        flat = Flat({'c': 1, 'd': {'foo': 2}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({'foo': 2}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({'foo': 2}, flat['d'])
        self.assertEqual(2, flat.d['foo'])
        self.assertEqual(2, flat['d']['foo'])
        self.assertEqual(2, flat['d.foo'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo

    def test_non_str_keys(self):
        flat = Flat({'c': 1, 'd': {3: 2}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({3: 2}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({3: 2}, flat['d'])
        self.assertEqual(2, flat.d[3])
        with self.assertRaises(KeyError):
            _ = flat['d.3']

    def test_empty_dict(self):
        flat = Flat({'c': 1, 'd': {}})
        self.assertIsInstance(flat.d, dict)
        self.assertDictEqual({}, flat.d)
        self.assertIsInstance(flat['d'], dict)
        self.assertDictEqual({}, flat['d'])


class TestNested(unittest.TestCase):

    def test_str_keys(self):
        nested = Nested({'a': 'foo', 'b': {'c': 1, 'd': {'bar': 2}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({'bar': 2}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({'bar': 2}, nested['b']['d'])
        self.assertEqual(2, nested.b.d['bar'])
        self.assertEqual(2, nested['b']['d']['bar'])
        self.assertEqual(2, nested['b.d.bar'])
        with self.assertRaises(AttributeError):
            _ = nested.b.d.bar

    def test_non_str_keys(self):
        nested = Nested({'a': 'foo', 'b': {'c': 1, 'd': {3: 2}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({3: 2}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({3: 2}, nested['b']['d'])
        self.assertIsInstance(nested['b.d'], dict)
        self.assertDictEqual({3: 2}, nested['b.d'])
        self.assertEqual(2, nested.b.d[3])
        with self.assertRaises(KeyError):
            _ = nested['b.d.3']

    def test_empty_dict(self):
        nested = Nested({'a': 'foo', 'b': {'c': 1, 'd': {}}})
        self.assertIsInstance(nested.b.d, dict)
        self.assertDictEqual({}, nested.b.d)
        self.assertIsInstance(nested['b']['d'], dict)
        self.assertDictEqual({}, nested['b']['d'])
        self.assertIsInstance(nested['b.d'], dict)
        self.assertDictEqual({}, nested['b.d'])


if __name__ == '__main__':
    unittest.main()

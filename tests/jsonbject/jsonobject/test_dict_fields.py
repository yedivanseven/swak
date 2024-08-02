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
        self.assertDictEqual({}, flat.d)
        self.assertDictEqual({}, flat['d'])

    def test_str_keys(self):
        flat = Flat({'d': {'foo': 1}})
        self.assertDictEqual({'foo': 1}, flat.d)
        self.assertDictEqual({'foo': 1}, flat['d'])
        self.assertEqual(1, flat['d.foo'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo

    def test_mixed_keys(self):
        flat = Flat({'d': {2: 3, 'foo': 1}})
        self.assertDictEqual({2: 3, 'foo': 1}, flat.d)
        self.assertDictEqual({2: 3, 'foo': 1}, flat['d'])
        self.assertEqual(1, flat['d.foo'])
        with self.assertRaises(KeyError):
            _ = flat['d.2']
        with self.assertRaises(AttributeError):
            _ = flat.d.foo

    def test_merge_str_keys_into_str_keys(self):
        flat = Flat({'d': {'foo': 'bar', 'baz': 1}}, d={'baz': 2})
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat.d)
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertEqual(2, flat['d.baz'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo
        with self.assertRaises(AttributeError):
            _ = flat.d.baz

    def test_merge_str_keys_into_mixed_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={'foo': 3})
        self.assertDictEqual({1: 2, 'foo': 3}, flat.d)
        self.assertDictEqual({1: 2, 'foo': 3}, flat['d'])
        self.assertEqual(3, flat['d.foo'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo

    def test_dont_merge_non_str_keys_into_mixed_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={1: 3})
        self.assertDictEqual({1: 3}, flat.d)
        self.assertDictEqual({1: 3}, flat['d'])

    def test_deep_merge_str_keys_into_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 'world': 2}}},
            d={'baz': {'world': 3}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 1, 'world': 3}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        with self.assertRaises(AttributeError):
            _ = flat.d.foo
        with self.assertRaises(AttributeError):
            _ = flat.d.baz
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': 1, 'world': 3}, flat['d.baz'])

    def test_deep_merge_str_keys_into_mixed_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {'hello': 4}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 4, 3: 2}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])

    def test_dont_deep_merge_non_str_keys_into_mixed_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {3: 4}}
        )
        expected = {'foo': 'bar', 'baz': {3: 4}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])

    def test_dont_deep_merge_str_keys_into_mixed_keys(self):
        flat = Flat(
            {'d': {4: 'foo', 'bar': {'hello': 1, 'world': 2}}},
            d={'bar': {'world': 3}}
        )
        expected = {4: 'foo', 'bar': {'hello': 1, 'world': 3}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])

    def test_dont_deep_merge_non_str_keys_mixed_2(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 'world': 2}}},
            d={4: {'world': 3}}
        )
        expected = {4: {'world': 3}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])

    def test_dont_deep_merge_non_str_keys_mixed_3(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {3: 4}}
        )
        expected = {'foo': 'bar', 'baz': {3: 4}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])


# ToDo. We're not even here yet!
class TestNested(unittest.TestCase):

    def test_empty(self):
        nested = Nested({'b': {'d': {}}})
        self.assertDictEqual({}, nested.b.d)
        self.assertDictEqual({}, nested.b['d'])
        self.assertDictEqual({}, nested['b']['d'])
        self.assertDictEqual({}, nested['b.d'])

    def test_str_keys_only(self):
        nested = Nested({'b': {'d': {'foo': 1, 'bar': 2}}})
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested.b.d)
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested.b['d'])
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested['b.d'])
        self.assertEqual(1, nested['b.d.foo'])
        self.assertEqual(2, nested['b.d.bar'])
        with self.assertRaises(AttributeError):
            _ = nested.b.d.foo
        with self.assertRaises(AttributeError):
            _ = nested.b.d.bar

    def test_str_keys_mixed(self):
        nested = Nested({'b': {'d': {1: 2, 'foo': 'bar'}}})
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b.d)
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b']['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])
        with self.assertRaises(AttributeError):
            _ = nested.b.d.foo
        with self.assertRaises(KeyError):
            _ = nested['b.d.1']

    def test_deep_merge_string_keys_only(self):
        nested = Nested(
            {'b': {'d': {'foo': 1, 'bar': 2}}},
            b={'d': {'bar': 3}}
        )
        self.assertDictEqual({'foo': 1, 'bar': 3}, nested.b.d)
        self.assertDictEqual({'foo': 1, 'bar': 3}, nested.b['d'])
        self.assertDictEqual({'foo': 1, 'bar': 3}, nested['b']['d'])
        self.assertDictEqual({'foo': 1, 'bar': 3}, nested['b.d'])
        self.assertEqual(1, nested['b.d.foo'])
        self.assertEqual(3, nested['b.d.bar'])
        with self.assertRaises(AttributeError):
            _ = nested.b.d.foo
        with self.assertRaises(AttributeError):
            _ = nested.b.d.bar


if __name__ == '__main__':
    unittest.main()

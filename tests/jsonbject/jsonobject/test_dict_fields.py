import unittest
from swak.jsonobject import JsonObject


class Flat(JsonObject):
    c: int = 42
    d: dict


class Nested(JsonObject):
    a: str = 'foo'
    b: Flat


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    pass


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

    def test_mixed_keys(self):
        flat = Flat({'d': {2: 3, 'foo': 1}})
        self.assertDictEqual({2: 3, 'foo': 1}, flat.d)
        self.assertDictEqual({2: 3, 'foo': 1}, flat['d'])
        self.assertEqual(1, flat['d.foo'])

    def test_append_empty(self):
        flat = Flat({'d': {}}, d={'foo': 'bar', 'baz': 2})
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat.d)
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat['d'])
        self.assertEqual('bar', flat['d.foo'])

    def test_merge_str_keys_into_str_keys(self):
        flat = Flat({'d': {'foo': 'bar', 'baz': 1}}, d={'baz': 2})
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat.d)
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, flat['d'])
        self.assertEqual('bar', flat['d.foo'])

    def test_merge_str_keys_into_mixed_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={'foo': 3})
        self.assertDictEqual({1: 2, 'foo': 3}, flat.d)
        self.assertDictEqual({1: 2, 'foo': 3}, flat['d'])
        self.assertEqual(3, flat['d.foo'])

    def test_merge_non_str_keys_into_mixed_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={1: 3})
        self.assertDictEqual({1: 3, 'foo': 'bar'}, flat.d)
        self.assertDictEqual({1: 3, 'foo': 'bar'}, flat['d'])
        self.assertEqual('bar', flat['d.foo'])

    def test_nest_str_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={'foo': {1: 3}})
        self.assertDictEqual({1: 2, 'foo': {1: 3}}, flat.d)
        self.assertDictEqual({1: 2, 'foo': {1: 3}}, flat['d'])
        self.assertDictEqual({1: 3}, flat['d.foo'])

    def test_nest_non_str_keys(self):
        flat = Flat({'d': {1: 2, 'foo': 'bar'}}, d={1: {'baz': 3}})
        self.assertDictEqual({1: {'baz': 3}, 'foo': 'bar'}, flat.d)
        self.assertDictEqual({1: {'baz': 3}, 'foo': 'bar'}, flat['d'])
        self.assertEqual('bar', flat['d.foo'])

    def test_deep_append_empty(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {}}},
            d={'baz': {'world': 3}}
        )
        expected = {'foo': 'bar', 'baz': {'world': 3}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertDictEqual({'world': 3}, flat['d.baz'])

    def test_deep_merge_str_keys_into_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 'world': 2}}},
            d={'baz': {'world': 3}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 1, 'world': 3}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': 1, 'world': 3}, flat['d.baz'])
        self.assertEqual(1, flat['d.baz.hello'])
        self.assertEqual(3, flat['d.baz.world'])

    def test_deep_merge_str_keys_into_mixed_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {'hello': 4}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 4, 3: 2}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': 4, 3: 2}, flat['d.baz'])
        self.assertEqual(4, flat['d.baz.hello'])

    def test_deep_merge_non_str_keys_into_mixed_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {3: 4}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 1, 3: 4}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': 1, 3: 4}, flat['d.baz'])
        self.assertEqual(1, flat['d.baz.hello'])

    def test_deep_nest_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {'hello': {'world': 42}}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': {'world': 42}, 3: 2}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': {'world': 42}, 3: 2}, flat['d.baz'])
        self.assertDictEqual({'world': 42}, flat['d.baz.hello'])

    def test_deep_nest_non_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': {3: {'world': 42}}}
        )
        expected = {'foo': 'bar', 'baz': {'hello': 1, 3: {'world': 42}}}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertDictEqual({'hello': 1, 3: {'world': 42}}, flat['d.baz'])
        self.assertEqual(1, flat['d.baz.hello'])

    def test_deep_collapse_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 'baz': {'hello': 1, 3: 2}}},
            d={'baz': 4}
        )
        expected = {'foo': 'bar', 'baz': 4}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])
        self.assertEqual(4, flat['d.baz'])

    def test_deep_collapse_non_str_keys(self):
        flat = Flat(
            {'d': {'foo': 'bar', 1: {'hello': 2, 3: 4}}},
            d={1: 42}
        )
        expected = {'foo': 'bar', 1: 42}
        self.assertDictEqual(expected, flat.d)
        self.assertDictEqual(expected, flat['d'])
        self.assertEqual('bar', flat['d.foo'])


class TestNested(unittest.TestCase):

    def test_empty(self):
        nested = Nested({'b': {'d': {}}})
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({}, nested.b.d)
        self.assertDictEqual({}, nested.b['d'])
        self.assertDictEqual({}, nested['b']['d'])
        self.assertDictEqual({}, nested['b.d'])

    def test_str_keys(self):
        nested = Nested({'b': {'d': {'foo': 1, 'bar': 2}}})
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested.b.d)
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested.b['d'])
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 1, 'bar': 2}, nested['b.d'])
        self.assertEqual(1, nested['b.d.foo'])
        self.assertEqual(2, nested['b.d.bar'])

    def test_mixed_keys(self):
        nested = Nested({'b': {'d': {1: 2, 'foo': 'bar'}}})
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b.d)
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b']['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])

    def test_append_empty(self):
        nested = Nested({'b': {'d': {}}}, b={'d': {1: 2, 'foo': 'bar'}})
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b.d)
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested.b['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b']['d'])
        self.assertDictEqual({1: 2, 'foo': 'bar'}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])

    def test_merge_str_keys_into_str_keys(self):
        nested = Nested(
            {'b': {'d': {'foo': 'bar', 'baz': 1}}},
            b={'d': {'baz': 2}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, nested.b.d)
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, nested.b['d'])
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 'bar', 'baz': 2}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])
        self.assertEqual(2, nested['b.d.baz'])

    def test_merge_str_keys_into_mixed_keys(self):
        nested = Nested(
            {'b': {'d': {1: 2, 'foo': 'bar'}}},
            b={'d': {'foo': 3}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 3, 1: 2}, nested.b.d)
        self.assertDictEqual({'foo': 3, 1: 2}, nested.b['d'])
        self.assertDictEqual({'foo': 3, 1: 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 3, 1: 2}, nested['b.d'])
        self.assertEqual(3, nested['b.d.foo'])

    def test_merge_non_str_keys_into_mixed_keys(self):
        nested = Nested({'b': {'d': {1: 2, 'foo': 'bar'}}}, b={'d': {1: 3}})
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 'bar', 1: 3}, nested.b.d)
        self.assertDictEqual({'foo': 'bar', 1: 3}, nested.b['d'])
        self.assertDictEqual({'foo': 'bar', 1: 3}, nested['b']['d'])
        self.assertDictEqual({'foo': 'bar', 1: 3}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])

    def test_nest_str_keys(self):
        nested = Nested(
            {'b': {'d': {1: 2, 'foo': 'bar'}}},
            b={'d': {'foo': {1: 3}}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': {1: 3}, 1: 2}, nested.b.d)
        self.assertDictEqual({'foo': {1: 3}, 1: 2}, nested.b['d'])
        self.assertDictEqual({'foo': {1: 3}, 1: 2}, nested['b']['d'])
        self.assertDictEqual({'foo': {1: 3}, 1: 2}, nested['b.d'])
        self.assertDictEqual({1: 3}, nested['b.d.foo'])

    def test_nest_non_str_keys(self):
        nested = Nested(
            {'b': {'d': {1: 2, 'foo': 'bar'}}},
            b={'d': {1: {'baz': 3}}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 'bar', 1: {'baz': 3}}, nested.b.d)
        self.assertDictEqual({'foo': 'bar', 1: {'baz': 3}}, nested.b['d'])
        self.assertDictEqual({'foo': 'bar', 1: {'baz': 3}}, nested['b']['d'])
        self.assertDictEqual({'foo': 'bar', 1: {'baz': 3}}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])

    def test_collapse_str_keys(self):
        nested = Nested(
            {'b': {'d': {1: 2, 'foo': {1: 3}}}},
            b={'d': {'foo': 'bar'}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested.b.d)
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested.b['d'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])

    def test_collapse_non_str_keys(self):
        nested = Nested(
            {'b': {'d': {1: {'baz': 3}, 'foo': 'bar'}}},
            b={'d': {1: 2}}
        )
        self.assertEqual('foo', nested['a'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested.b.d)
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested.b['d'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested['b']['d'])
        self.assertDictEqual({'foo': 'bar', 1: 2}, nested['b.d'])
        self.assertEqual('bar', nested['b.d.foo'])


class TestUpdateFlat(unittest.TestCase):

    def setUp(self):
        self.flat = Flat(d={1: 2, 'foo': 3})

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'c'))
        self.assertIsInstance(obj.c, int)
        self.assertEqual(42, obj.c)
        self.assertTrue(hasattr(obj, 'd'))

    def test_str_keys(self):
        updated = self.flat(d={'foo': 'bar'})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': 'bar'}, updated.d)

    def test_non_str_keys(self):
        updated = self.flat(d={1: 4})
        self.check_attributes(updated)
        self.assertDictEqual({1: 4, 'foo': 3}, updated.d)

    def test_dont_append(self):
        updated = self.flat(d={'baz': 4})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': 3}, updated.d)

    def test_nest(self):
        updated = self.flat(d={'foo': {'bar': 4}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': {'bar': 4}}, updated.d)

    def test_collapse(self):
        flat = Flat(d={1: 2, 'foo': {3: 4}})
        updated = flat(d={1: 2, 'foo': 3})
        self.assertDictEqual({1: 2, 'foo': 3}, updated.d)


class TestUpdateNested(unittest.TestCase):

    def setUp(self):
        self.nested = Nested(b={'d': {1: 2, 'foo': 3}})

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, Flat)
        self.assertTrue(hasattr(obj.b, 'c'))
        self.assertIsInstance(obj.b.c, int)
        self.assertEqual(42, obj.b.c)
        self.assertTrue(hasattr(obj.b, 'd'))

    def test_str_keys(self):
        updated = self.nested(b={'d': {'foo': 'bar'}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': 'bar'}, updated.b.d)

    def test_non_str_keys(self):
        updated = self.nested(b={'d': {1: 4}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 4, 'foo': 3}, updated.b.d)

    def test_dont_append(self):
        updated = self.nested(b={'d': {'baz': 4}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': 3}, updated.b.d)

    def test_nest(self):
        updated = self.nested(b={'d': {'foo': {'bar': 4}}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': {'bar': 4}}, updated.b.d)

    def test_collapse(self):
        nested = Nested(b={'d': {1: 2, 'foo': {'bar': 4}}})
        updated = nested(b={'d': {'foo': 3}})
        self.check_attributes(updated)
        self.assertDictEqual({1: 2, 'foo': 3}, updated.b.d)


class TestExtra(unittest.TestCase):

    def test_initialization(self):
        extra = Extra({'d': {1: 2}})
        self.assertDictEqual({1: 2}, extra.d)

    def test_append(self):
        extra = Extra({'d': {1: 2}}, d={'foo': 'bar'})
        self.assertDictEqual({1: 2, 'foo': 'bar'}, extra.d)

    def test_merge(self):
        extra = Extra({'d': {1: 2, 'foo': 'bar'}}, d={1: 42})
        self.assertDictEqual({1: 42, 'foo': 'bar'}, extra.d)

    def test_collapse(self):
        extra = Extra({'d': {1: 2, 'foo': {3: 4}}}, d={'foo': 'bar'})
        self.assertDictEqual({1: 2, 'foo': 'bar'}, extra.d)

    def test_update_dont_append(self):
        extra = Extra({'d': {1: 2}})
        updated = extra(d={'foo': 'bar'})
        self.assertDictEqual({1: 2}, updated.d)

    def test_update_merge(self):
        extra = Extra({'d': {1: 2, 'foo': 'bar'}})
        updated = extra(d={1: 42})
        self.assertDictEqual({1: 42, 'foo': 'bar'}, updated.d)

    def test_update_collapse(self):
        extra = Extra({'d': {1: 2, 'foo': {3: 4}}})
        updated = extra(d={'foo': 'bar'})
        self.assertDictEqual({1: 2, 'foo': 'bar'}, updated.d)


if __name__ == '__main__':
    unittest.main()

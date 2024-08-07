import unittest
import pandas as pd
from swak.jsonobject import JsonObject, JsonObjects


class Item(JsonObject):
    a: int = 1
    b: str = 'foo'


class Items(JsonObjects, item_type=Item):
    pass


class CustomType:

    def __init__(self, *args, **kwargs):
        pass

    @property
    def as_json(self):
        return 'as json'

    @property
    def as_dtype(self):
        return 'as dtype'


class CustomItem(Item):
    c: CustomType = CustomType()


class CustomItems(JsonObjects, item_type=CustomItem):
    pass


class NestedItem(JsonObject):
    c: float = 1.0
    d: Item = Item()


class NestedItems(JsonObjects, item_type=NestedItem):
    pass


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    a: int = 1
    b: str = 'foo'


class Extras(JsonObjects, item_type=Extra):
    pass


class NestedExtra(JsonObject):
    c: float = 1.0
    d: Extra = Extra()


class NestedExtras(JsonObjects, item_type=NestedExtra):
    pass


class TestMagic(unittest.TestCase):

    def setUp(self) -> None:
        self.items = Items([Item(), Item(a=2, b='bar')])
        self.custom = CustomItems([CustomItem(), CustomItem(a=2, b='bar')])

    def test_str(self):
        expected = '[{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}]'
        self.assertEqual(expected, str(self.items))

    def test_custom_str(self):
        expected = ('[{"a": 1, "b": "foo", "c": "as json"},'
                    ' {"a": 2, "b": "bar", "c": "as json"}]')
        self.assertEqual(expected, str(self.custom))

    def test_iter(self):
        expected = [{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}]
        for i, item in enumerate(iter(self.items)):
            self.assertDictEqual(expected[i], item.as_json)

    def test_star_expansion(self):
        expected = [Item({"a": 1, "b": "foo"}), Item({"a": 2, "b": "bar"})]
        actual = [*self.items]
        self.assertListEqual(expected, actual)

    def test_reversed_expansion(self):
        expected = [{"a": 2, "b": "bar"}, {"a": 1, "b": "foo"}]
        for i, item in enumerate(reversed(self.items)):
            self.assertDictEqual(expected[i], item.as_json)

    def test_len(self):
        self.assertEqual(0, len(Items()))
        self.assertEqual(2, len(self.items))

    def test_getattr(self):
        column = self.items.a
        self.assertIsInstance(column, list)
        self.assertListEqual([1, 2], column)

    def test_getattr_raises_keys(self):
        with self.assertRaises(AttributeError):
            _ = self.items.keys()

    def test_getattr_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            _ = self.items.foo

    def test_getitem_int_index(self):
        item = self.items[-1]
        expected = {"a": 2, "b": "bar"}
        self.assertIsInstance(item, Item)
        self.assertDictEqual(expected, item.as_json)

    def test_getitem_int_raises(self):
        with self.assertRaises(IndexError):
            _ = self.items[10]

    def test_getitem_slice(self):
        items = Items([Item(), Item(a=2, b='bar'), Item(a=3, b='baz')])
        expected = [{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}]
        actual: JsonObjects = items[:2]
        self.assertIsInstance(actual, JsonObjects)
        self.assertListEqual(expected, actual.as_json)

    def test_getitem_str_index(self):
        column = self.items['b']
        expected = ['foo', 'bar']
        self.assertIsInstance(column, list)
        self.assertListEqual(expected, column)

    def test_getitem_str_raises(self):
        with self.assertRaises(KeyError):
            _ = self.items['c']

    def test_getitem_nested_str_index(self):
        items = NestedItems([
            NestedItem(),
            NestedItem(d={"a": 2, "b": "bar"}),
            NestedItem(d={"a": 3, "b": "baz"})
        ])
        column = items['d.b']
        expected = ['foo', 'bar', 'baz']
        self.assertIsInstance(column, list)
        self.assertListEqual(expected, column)

    def test_bool_true(self):
        self.assertTrue(self.items)

    def test_bool_false(self):
        self.assertFalse(Items())

    def test_contains_true_jsonobject(self):
        self.assertIn(Item(), self.items)

    def test_contains_true_dict(self):
        self.assertIn({'a': 1, 'b': 'foo'}, self.items)

    def test_contains_true_str(self):
        self.assertIn("{'a': 1, 'b': 'foo'}", self.items)

    def test_contains_true_json(self):
        self.assertIn('{"a": 1, "b": "foo"}', self.items)

    def test_contains_true_bytes(self):
        self.assertIn('{"a": 1, "b": "foo"}'.encode(), self.items)

    def test_contains_true_series(self):
        self.assertIn(pd.Series({'a': 1, 'b': 'foo'}), self.items)

    def test_contains_false_jsonobject(self):
        self.assertNotIn(Item(a=4, b='hello world'), self.items)

    def test_contains_false_dict(self):
        self.assertNotIn({'a': 4, 'b': 'hello world'}, self.items)

    def test_contains_false_json(self):
        self.assertNotIn('{"a": 4, "b": "hello world"}', self.items)

    def test_contains_false_not_json(self):
        self.assertNotIn(1, self.items)

    def test_contains_false_wrong_json(self):
        self.assertNotIn({'a': 'bar', 'b': 'foo'}, self.items)

    def test_equality_other(self):
        other = Items([Item(), Item(a=2, b='bar')])
        self.assertEqual(self.items, other)

    def test_equality_self(self):
        self.assertEqual(self.items, self.items)

    def test_equality_false_wrong_type(self):
        self.assertFalse(self.items == 1)

    def test_equality_false_wrong_content(self):
        other = Items([Item(), Item(a=2, b='baz')])
        self.assertFalse(self.items == other)

    def test_inequality_other_type(self):
        self.assertNotEqual(self.items, 1)

    def test_inequality_other_content(self):
        other = Items([Item(), Item(a=3, b='baz')])
        self.assertNotEqual(self.items, other)

    def test_inequality_false_other(self):
        other = Items([Item(), Item(a=2, b='bar')])
        self.assertFalse(self.items != other)

    def test_inequality_false_self(self):
        self.assertFalse(self.items != self.items)

    def test_callable(self):
        self.assertTrue(callable(self.items))

    def test_call_flat(self):
        expected = [{'a': 3, 'b': 'baz'}, {'a': 3, 'b': 'baz'}]
        updated = self.items({'a': 3}, b='baz')
        self.assertListEqual(expected, updated.as_json)

    def test_call_nested_dict(self):
        items = NestedItems([
            NestedItem(),
            NestedItem(d={"a": 2, "b": "bar"}),
            NestedItem(d={"a": 3, "b": "baz"})
        ])
        expected = [
            {"c": 1.0, "d": {"a": 3, "b": "hello world"}},
            {"c": 1.0, "d": {"a": 3, "b": "hello world"}},
            {"c": 1.0, "d": {"a": 3, "b": "hello world"}}
        ]
        updated = items({'d': {'a': 3}}, d={'b': "hello world"})
        self.assertListEqual(expected, updated.as_json)

    def test_call_nested_dot_key(self):
        items = NestedItems([
            NestedItem(),
            NestedItem(d={"a": 2, "b": "bar"}),
            NestedItem(d={"a": 3, "b": "baz"})
        ])
        expected = [
            {"c": 1.0, "d": {"a": 4, "b": "hello world"}},
            {"c": 1.0, "d": {"a": 4, "b": "hello world"}},
            {"c": 1.0, "d": {"a": 4, "b": "hello world"}}
        ]
        updated = items({'d.b': 'hello world'}, d={'a': 4})
        self.assertListEqual(expected, updated.as_json)


class TestAddition(unittest.TestCase):

    def setUp(self) -> None:
        self.items = Items([Item(), Item(a=2, b='bar')])
        self.expected = Items([Item(), Item(a=2, b='bar'), Item(a=3, b='baz')])

    def test_empty_string(self):
        actual = self.items + ''
        self.assertEqual(self.items, actual)

    def test_empty_list_string(self):
        actual = self.items + '[]'
        self.assertEqual(self.items, actual)

    def test_empty_list(self):
        actual = self.items + []
        self.assertEqual(self.items, actual)

    def test_none(self):
        actual = self.items + None
        self.assertEqual(self.items, actual)

    def test_empty_jsonobjects(self):
        actual = self.items + Items()
        self.assertEqual(self.items, actual)

    def test_empty_dataframe(self):
        actual = self.items + pd.DataFrame()
        self.assertEqual(self.items, actual)

    def test_jsonobject(self):
        actual = self.items + Item(a=3, b='baz')
        self.assertEqual(self.expected, actual)

    def test_series(self):
        actual = self.items + pd.Series({'a': 3, 'b': 'baz'})
        self.assertEqual(self.expected, actual)

    def test_dict(self):
        actual = self.items + {'a': 3, 'b': 'baz'}
        self.assertEqual(self.expected, actual)

    def test_json(self):
        actual = self.items + '{"a": 3, "b": "baz"}'
        self.assertEqual(self.expected, actual)

    def test_list_of_jsonobject(self):
        actual = self.items + [Item(a=3, b='baz')]
        self.assertEqual(self.expected, actual)

    def test_list_of_series(self):
        actual = self.items + [pd.Series({'a': 3, 'b': 'baz'})]
        self.assertEqual(self.expected, actual)

    def test_list_of_dicts(self):
        actual = self.items + [{'a': 3, 'b': 'baz'}]
        self.assertEqual(self.expected, actual)

    def test_str_with_list_of_dicts(self):
        actual = self.items + '[{"a": 3, "b": "baz"}]'
        self.assertEqual(self.expected, actual)

    def test_jsonobjects(self):
        actual = self.items + Items([{'a': 3, 'b': 'baz'}])
        self.assertEqual(self.expected, actual)

    def test_dataframe(self):
        actual = self.items + pd.DataFrame([{'a': 3, 'b': 'baz'}])
        self.assertEqual(self.expected, actual)


class TestRightAddition(unittest.TestCase):

    def setUp(self) -> None:
        self.items = Items([Item(), Item(a=2, b='bar')])
        self.expected = Items([Item(a=3, b='baz'), Item(), Item(a=2, b='bar')])

    def test_empty_string(self):
        actual = '' + self.items
        self.assertEqual(self.items, actual)

    def test_empty_list_string(self):
        actual = '[]' + self.items
        self.assertEqual(self.items, actual)

    def test_empty_list(self):
        actual = [] + self.items
        self.assertEqual(self.items, actual)

    def test_none(self):
        actual = None + self.items
        self.assertEqual(self.items, actual)

    def test_empty_jsonobjects(self):
        actual = Items() + self.items
        self.assertEqual(self.items, actual)

    def test_jsonobject(self):
        actual = Item(a=3, b='baz') + self.items
        self.assertEqual(self.expected, actual)

    def test_series_raises(self):
        with self.assertRaises(TypeError):
            _ = pd.Series({'a': 3, 'b': 'baz'}) + self.items

    def test_dict(self):
        actual = {'a': 3, 'b': 'baz'} + self.items
        self.assertEqual(self.expected, actual)

    def test_json(self):
        actual = '{"a": 3, "b": "baz"}' + self.items
        self.assertEqual(self.expected, actual)

    def test_list_of_jsonobject(self):
        actual = [Item(a=3, b='baz')] + self.items
        self.assertEqual(self.expected, actual)

    def test_list_of_series(self):
        actual = [pd.Series({'a': 3, 'b': 'baz'})] + self.items
        self.assertEqual(self.expected, actual)

    def test_list_of_dicts(self):
        actual = [{'a': 3, 'b': 'baz'}] + self.items
        self.assertEqual(self.expected, actual)

    def test_str_with_list_of_dicts(self):
        actual = '[{"a": 3, "b": "baz"}]' + self.items
        self.assertEqual(self.expected, actual)

    def test_jsonobjects(self):
        actual = Items([{'a': 3, 'b': 'baz'}]) + self.items
        self.assertEqual(self.expected, actual)

    def test_dataframe_raises(self):
        with self.assertRaises(TypeError):
            _ = pd.DataFrame([{'a': 3, 'b': 'baz'}]) + self.items


class TestExtra(unittest.TestCase):

    def test_getattr(self):
        extras = Extras([{'c': 'bar'}, {'d': 'baz'}])
        self.assertListEqual([1, 1], extras.a)
        self.assertListEqual(['foo', 'foo'], extras.b)
        self.assertListEqual(['bar', None], extras.c)
        self.assertListEqual([None, 'baz'], extras.d)
        with self.assertRaises(AttributeError):
            _ = extras.e

    def test_getitem(self):
        extras = Extras([{'c': 'bar'}, {'d': 'baz'}])
        self.assertListEqual([1, 1], extras['a'])
        self.assertListEqual(['foo', 'foo'], extras['b'])
        self.assertListEqual(['bar', None], extras['c'])
        self.assertListEqual([None, 'baz'], extras['d'])
        with self.assertRaises(KeyError):
            _ = extras['e']

    def test_getitem_nested(self):
        extras = NestedExtras([{'d': {'e': 'bar'}}, {'d': {'f': 'baz'}}])
        self.assertListEqual([1, 1], extras['d.a'])
        self.assertListEqual(['foo', 'foo'], extras['d.b'])
        self.assertListEqual(['bar', None], extras['d.e'])
        self.assertListEqual([None, 'baz'], extras['d.f'])
        with self.assertRaises(KeyError):
            _ = extras['d.g']

    def test_update(self):
        extras = Extras([{'c': 'bar'}, {'d': 'baz'}])
        updated = extras({'c': 'hello', 'e': 'world'}, d='answer', f=42)
        expected = [
            {'a': 1, 'b': 'foo', 'c': 'hello'},
            {'a': 1, 'b': 'foo', 'd': 'answer'}
        ]
        self.assertListEqual(expected, updated.as_json)

    def test_update_nested(self):
        extras = NestedExtras([{'d': {'e': 'bar'}}, {'d': {'f': 'baz'}}])
        updated = extras({'d.e': 'answer'}, d={'f': 42})
        expected = [
            {'c': 1.0, 'd': {'a': 1, 'b': 'foo', 'e': 'answer'}},
            {'c': 1.0, 'd': {'a': 1, 'b': 'foo', 'f': 42}}
        ]
        self.assertListEqual(expected, updated.as_json)


if __name__ == '__main__':
    unittest.main()

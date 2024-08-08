import unittest
import pandas as pd
from swak.jsonobject import JsonObject, JsonObjects


class Empty(JsonObject):
    pass


class Empties(JsonObjects, item_type=Empty):
    pass


class Item(JsonObject):
    a: int = 1
    b: str = 'foo'


class Items(JsonObjects, item_type=Item):
    pass


class CustomType:

    def __init__(self, *args, **kwargs):
        pass

    @property
    def as_json(self) -> str:
        return 'as json'

    @property
    def as_dtype(self) -> str:
        return 'as dtype'


class CustomItem(Item):
    c: CustomType = CustomType()

    @property
    def p(self):
        return 'property'

    @classmethod
    def cls(cls):
        pass

    @staticmethod
    def stat():
        pass

    def meth(self):
        pass


class CustomItems(JsonObjects, item_type=CustomItem):

    @property
    def as_dtype(self) -> list:
        return self.as_json


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    a: int = 1
    b: str = 'foo'


class Extras(JsonObjects, item_type=Extra):
    pass


class TestAttributes(unittest.TestCase):

    def setUp(self) -> None:
        self.items = [Item(), Item()]
        self.custom_items = [CustomItem(), CustomItem()]

    def test_as_json(self):
        items = Items(self.items)
        expected = [{'a': 1, 'b': 'foo'}, {'a': 1, 'b': 'foo'}]
        self.assertTrue(hasattr(items, 'as_json'))
        self.assertIsInstance(items.as_json, list)
        self.assertListEqual(expected, items.as_json)

    def test_instantiation_from_as_json(self):
        items = Items(self.items)
        self.assertListEqual(items.as_json, Items(items.as_json).as_json)

    def test_custom_as_json(self):
        items = CustomItems(self.custom_items)
        expected = [
            {'a': 1, 'b': 'foo', 'c': 'as json'},
            {'a': 1, 'b': 'foo', 'c': 'as json'}
        ]
        self.assertTrue(hasattr(items, 'as_json'))
        self.assertIsInstance(items.as_json, list)
        self.assertListEqual(expected, items.as_json)

    def test_instantiation_from_custom_as_json(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual(items.as_json, CustomItems(items.as_json).as_json)

    def test_as_dtype(self):
        items = Items(self.items)
        expected = '[{"a": 1, "b": "foo"}, {"a": 1, "b": "foo"}]'
        self.assertTrue(hasattr(items, 'as_dtype'))
        self.assertEqual(expected, items.as_dtype)

    def test_instantiation_from_as_dtype(self):
        items = Items(self.items)
        self.assertListEqual(items.as_json, Items(items.as_dtype).as_json)

    def test_custom_as_dtype(self):
        items = CustomItems(self.custom_items)
        expected = [
            {'a': 1, 'b': 'foo', 'c': 'as json'},
            {'a': 1, 'b': 'foo', 'c': 'as json'}
        ]
        self.assertTrue(hasattr(items, 'as_dtype'))
        self.assertIsInstance(items.as_dtype, list)
        self.assertListEqual(expected, items.as_dtype)

    def test_instantiation_from_custom_as_dtype(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual(items.as_json, CustomItems(items.as_dtype).as_json)

    def test_as_df(self):
        items = Items(self.items)
        expected = pd.DataFrame([{'a': 1, 'b': 'foo'}, {'a': 1, 'b': 'foo'}])
        expected.columns.name = 'Item'
        self.assertTrue(hasattr(items, 'as_df'))
        self.assertIsInstance(items.as_df, pd.DataFrame)
        pd.testing.assert_frame_equal(expected, items.as_df)

    def test_instantiation_from_as_df(self):
        items = Items(self.items)
        pd.testing.assert_frame_equal(items.as_df, Items(items.as_df).as_df)

    def test_custom_as_df(self):
        items = CustomItems(self.custom_items)
        expected = pd.DataFrame([
            {'a': 1, 'b': 'foo', 'c': 'as dtype'},
            {'a': 1, 'b': 'foo', 'c': 'as dtype'}
        ])
        expected.columns.name = 'CustomItem'
        self.assertTrue(hasattr(items, 'as_df'))
        self.assertIsInstance(items.as_df, pd.DataFrame)
        pd.testing.assert_frame_equal(expected, items.as_df)

    def test_instantiation_from_custom_as_df(self):
        items = CustomItems(self.custom_items)
        pd.testing.assert_frame_equal(
            items.as_df,
            CustomItems(items.as_df).as_df
        )

    def test_empty_as_df(self):
        items = Items()
        self.assertTrue(hasattr(items, 'as_df'))
        self.assertIsInstance(items.as_df, pd.DataFrame)
        expected = pd.DataFrame([], columns=['a', 'b'])
        expected.columns.name = 'Item'
        pd.testing.assert_frame_equal(expected, items.as_df)

    def test_empty_empty_as_df(self):
        empties = Empties()
        self.assertTrue(hasattr(empties, 'as_df'))
        self.assertIsInstance(empties.as_df, pd.DataFrame)
        expected = pd.DataFrame([], columns=[])
        expected.columns.name = 'Empty'
        pd.testing.assert_frame_equal(expected, empties.as_df)

    def test_extra_df(self):
        extras = Extras([{'c': 'bar'}, {'d': 'baz'}])
        self.assertTrue(hasattr(extras, 'as_df'))
        self.assertIsInstance(extras.as_df, pd.DataFrame)
        data = [
            {'a': 1, 'b': 'foo', 'c': 'bar'},
            {'a': 1, 'b': 'foo', 'd': 'baz'},
        ]
        expected = pd.DataFrame(data, columns=['a', 'b', 'c', 'd'])
        expected.columns.name = 'Extra'
        pd.testing.assert_frame_equal(expected, extras.as_df)

    def test_getattr_property(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual(['property', 'property'], items.p)

    def test_getattr_classmethod(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual([CustomItem.cls, CustomItem.cls], items.cls)

    def test_getattr_staticmethod(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual([CustomItem.stat, CustomItem.stat], items.stat)

    def test_getattr_method(self):
        items = CustomItems(self.custom_items)
        self.assertListEqual([items[0].meth, items[1].meth], items.meth)


if __name__ == '__main__':
    unittest.main()

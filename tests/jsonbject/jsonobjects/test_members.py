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

    @property
    def as_dtype(self):
        return self.as_json


class TestAttributes(unittest.TestCase):

    def setUp(self) -> None:
        self.items = [Item(), Item()]
        self.custom_items = [CustomItem(), CustomItems()]

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
        self.assertIsInstance(items.as_dtype, str)
        self.assertEqual(expected, items.as_dtype)

    def test_custom_as_dtype(self):
        items = CustomItems(self.custom_items)
        expected = [
            {'a': 1, 'b': 'foo', 'c': 'as json'},
            {'a': 1, 'b': 'foo', 'c': 'as json'}
        ]
        self.assertTrue(hasattr(items, 'as_dtype'))
        self.assertIsInstance(items.as_dtype, list)
        self.assertListEqual(expected, items.as_dtype)

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


if __name__ == '__main__':
    unittest.main()

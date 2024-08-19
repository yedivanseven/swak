import unittest
from swak.jsonobject import JsonObjects, JsonObject
from swak.jsonobject.exceptions import SchemaError
from swak.jsonobject.jsonobject import SchemaMeta


class Item(JsonObject):
    a: int = 1
    b: str = 'foo'


class TestSubclassing(unittest.TestCase):

    def test_intended_singular(self):

        class Intended(JsonObjects, item_type=Item):
            pass

        self.assertTrue(hasattr(Intended, '__item_type__'))
        self.assertIs(Intended.__item_type__, Item)

    def test_inherited_singular(self):

        class Intended(JsonObjects, item_type=Item):
            pass

        class Inherited(Intended):
            pass

        self.assertTrue(hasattr(Inherited, '__item_type__'))
        self.assertIs(Inherited.__item_type__, Item)

    def test_raises_on_missing_item_type(self):
        with self.assertRaises(SchemaError):

            class Wrong(JsonObjects):
                pass

    def test_raises_on_item_type_in_class_body(self):
        with self.assertRaises(SchemaError):

            class Wrong(JsonObjects):
                __item_type__ = Item

    def test_raises_on_item_type_not_schema_meta(self):
        with self.assertRaises(SchemaError):
            class Wrong(JsonObjects, item_type=int):
                pass

    def test_raises_on_item_type_not_jsonobject(self):

        class Wrong(metaclass=SchemaMeta):
            pass

        with self.assertRaises(SchemaError):

            class StillWrong(JsonObjects, item_type=Wrong):
                pass


if __name__ == '__main__':
    unittest.main()

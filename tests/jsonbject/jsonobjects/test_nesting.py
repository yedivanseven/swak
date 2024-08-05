import unittest
from swak.jsonobject import JsonObject, JsonObjects


class Item(JsonObject):
    a: int
    b: str


class Items(JsonObjects, item_type=Item):
    pass


class Child(Item):
    c: Items


class TestNesting(unittest.TestCase):

    def test_child_of_jsonobject(self):
        items = Items([{'a': 2, 'b': 'bar'}, {'a': 3, 'b': 'baz'}])
        item = {
            'a': 1,
            'b': 'foo',
            'c': items
        }
        child = Child(item)
        self.assertTrue(hasattr(child, 'c'))
        self.assertIsInstance(child.c, Items)
        self.assertEqual(items, child.c)


if __name__ == '__main__':
    unittest.main()

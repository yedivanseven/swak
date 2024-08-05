import unittest
from swak.jsonobject import JsonObject


class Child(JsonObject):
    c: int
    d: str


class Parent(JsonObject):
    b: int
    child: Child


class Grand(JsonObject):
    a: int
    parent: Parent


class TestGetItem(unittest.TestCase):

    def setUp(self) -> None:
        init = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        self.grand = Grand(init)
        self.parent = Parent(init['parent'])
        self.child = Child(init['parent']['child'])

    def test_root_leave(self):
        actual = self.grand['a']
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_root_node(self):
        actual = self.grand['parent']
        self.assertIsInstance(actual, Parent)
        self.assertEqual(self.parent, actual)

    def test_first_level_leave(self):
        actual = self.grand['parent.b']
        self.assertIsInstance(actual, int)
        self.assertEqual(self.parent['b'], actual)

    def test_first_level_node(self):
        actual = self.grand['parent.child']
        self.assertIsInstance(actual, Child)
        self.assertEqual(self.child, actual)

    def test_second_level_leave(self):
        actual = self.grand['parent.child.c']
        self.assertIsInstance(actual, int)
        self.assertEqual(self.child['c'], actual)

    def test_raises_missing_key(self):
        with self.assertRaises(AttributeError):
            _ = self.grand['parent.child.missing']


# Do we even need this method?
class TestGet(unittest.TestCase):

    def setUp(self) -> None:
        init = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        self.grand = Grand(init)
        self.parent = Parent(init['parent'])
        self.child = Child(init['parent']['child'])

    def test_root_leave(self):
        actual = self.grand.get('a')
        self.assertIsInstance(actual, int)
        self.assertEqual(1, actual)

    def test_root_node(self):
        actual = self.grand.get('parent')
        self.assertIsInstance(actual, Parent)
        self.assertEqual(self.parent, actual)

    def test_first_level_leave(self):
        actual = self.grand.get('parent.b')
        self.assertIsInstance(actual, int)
        self.assertEqual(self.parent['b'], actual)

    def test_first_level_node(self):
        actual = self.grand.get('parent.child')
        self.assertIsInstance(actual, Child)
        self.assertEqual(self.child, actual)

    def test_second_level_leave(self):
        actual = self.grand.get('parent.child.c')
        self.assertIsInstance(actual, int)
        self.assertEqual(self.child['c'], actual)

    def test_get_default_if_missing(self):
        actual = self.grand.get('parent.child.missing', 'foo')
        self.assertEqual('foo', actual)


if __name__ == '__main__':
    unittest.main()

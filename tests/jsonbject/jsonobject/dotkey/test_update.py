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


class AllowChild(Child, ignore_extra=False, raise_extra=False):
    pass


class AllowParent(Parent, ignore_extra=False, raise_extra=False):
    child: AllowChild


class AllowGrand(Grand):
    parent: AllowParent


class TestDictOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.grand = Grand({
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        })
        self.expected = {
            'a': 1,
            'parent': {
                'b': 4,
                'child': {
                    'c': 5,
                    'd': 'bar'
                }
            }
        }

    def test_fully_nested(self):
        updated = self.grand(self.expected)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_first_level_nested_flat_leave(self):
        update = {
            'parent.b': 4,
            'parent': {
                'child': {
                    'c': 5,
                    'd': 'bar'
                }
            }
        }
        updated = self.grand(update)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_first_level_nested_nested_leave(self):
        update = {
            'parent': {
                'b': 4
            },
            'parent.child': {
                'c': 5,
                'd': 'bar'
            }
        }
        updated = self.grand(update)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_first_level_nested_all_leaves(self):
        update = {
            'parent.b': 4,
            'parent.child': {
                'c': 5,
                'd': 'bar'
            }
        }
        updated = self.grand(update)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_all_levels_nested_one_leave(self):
        update = {
            'parent.b': 4,
            'parent.child': {
                'd': 'bar'
            },
            'parent.child.c': 5
        }
        updated = self.grand(update)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_all_levels_nested_all_leaves(self):
        update = {
            'parent.b': 4,
            'parent.child.d': 'bar',
            'parent.child.c': 5
        }
        updated = self.grand(update)
        self.assertDictEqual(self.expected, updated.as_json)


class TestDictAndKwargDict(unittest.TestCase):

    def setUp(self) -> None:
        self.grand = Grand({
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        })
        self.expected = {
            'a': 1,
            'parent': {
                'b': 4,
                'child': {
                    'c': 5,
                    'd': 'bar'
                }
            }
        }

    def test_fully_nested(self):
        child = {
            'child': {
                'c': 5,
                'd': 'bar'
            }
        }
        updated = self.grand({'parent': {'b': 4}}, parent=child)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_second_level_partially_nested(self):
        child = {
            'child.c': 5,
            'child': {
                'd': 'bar'
            }
        }
        updated = self.grand({'parent': {'b': 4}}, parent=child)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_second_level_flat(self):
        child = {
            'child.c': 5,
            'child.d': 'bar'
        }
        updated = self.grand({'parent': {'b': 4}}, parent=child)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_level_mix(self):
        child = {
            'child.d': 'bar'
        }
        updated = self.grand(
            {'parent': {'b': 4}, 'parent.child.c': 5},
            parent=child
        )
        self.assertDictEqual(self.expected, updated.as_json)

    def test_second_level_partially_nested_trumps_dict(self):
        child = {
            'child.c': 5,
            'child': {
                'd': 'bar'
            }
        }
        updated = self.grand(
            {'parent': {'b': 4}, 'parent.child.c': 6},
            parent=child
        )
        self.assertDictEqual(self.expected, updated.as_json)

    def test_second_level_fully_nested_trumps_dict(self):
        child = {
            'child.c': 5,
            'child.d': 'bar'
        }
        updated = self.grand(
            {'parent': {'b': 4}, 'parent.child.c': 6, 'parent.child.d': 'baz'},
            parent=child
        )
        self.assertDictEqual(self.expected, updated.as_json)


class TestKwargDictOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.grand = Grand({
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        })
        self.expected = {
            'a': 1,
            'parent': {
                'b': 4,
                'child': {
                    'c': 5,
                    'd': 'bar'
                }
            }
        }

    def test_fully_nested(self):
        parent = {'b': 4, 'child': {'c': 5, 'd': 'bar'}}
        updated = self.grand(parent=parent)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_partially_nested(self):
        parent = {'b': 4, 'child.c': 5, 'child': {'d': 'bar'}}
        updated = self.grand(parent=parent)
        self.assertDictEqual(self.expected, updated.as_json)

    def test_flat(self):
        parent = {'b': 4, 'child.c': 5, 'child.d': 'bar'}
        updated = self.grand(parent=parent)
        self.assertDictEqual(self.expected, updated.as_json)


class TestSequential(unittest.TestCase):

    def setUp(self) -> None:
        self.grand = Grand({
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        })
        self.expected = {
            'a': 1,
            'parent': {
                'b': 4,
                'child': {
                    'c': 5,
                    'd': 'bar'
                }
            }
        }

    def test_sequential_dict_one_level(self):
        updated = self.grand(
            {'parent.b': 4}
        )(
            {'parent': {'child.c': 5}}
        )(
            {'parent': {'child.d': 'bar'}}
        )
        self.assertDictEqual(self.expected, updated.as_json)

    def test_sequential_dict_all_levels(self):
        updated = self.grand(
            {'parent.b': 4}
        )(
            {'parent.child.c': 5}
        )(
            {'parent.child.d': 'bar'}
        )
        self.assertDictEqual(self.expected, updated.as_json)

    def test_sequential_kwarg(self):
        updated = self.grand(
            parent={'b': 4}
        )(
            parent={'child.c': 5}
        )(
            parent={'child.d': 'bar'}
        )
        self.assertDictEqual(self.expected, updated.as_json)


class TestAllowExtraFields(unittest.TestCase):

    def setUp(self) -> None:
        self.initial = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_first_level_dict(self):
        grand = AllowGrand(self.initial, parent={'e': 4})
        update = {'parent.e': 5}
        updated = grand(update)
        self.assertTrue(hasattr(updated.parent, 'e'))
        self.assertIsInstance(updated.parent.e, int)
        self.assertEqual(5, updated.parent.e)

    def test_first_level_kwarg(self):
        grand = AllowGrand(self.initial, parent={'e': 4})
        update = {'e': 5}
        updated = grand(parent=update)
        self.assertTrue(hasattr(updated.parent, 'e'))
        self.assertIsInstance(updated.parent.e, int)
        self.assertEqual(5, updated.parent.e)

    def test_first_level_str(self):
        grand = AllowGrand(self.initial, parent={'e': 4})
        update = '{"parent.e": 5}'
        updated = grand(update)
        self.assertTrue(hasattr(updated.parent, 'e'))
        self.assertIsInstance(updated.parent.e, int)
        self.assertEqual(5, updated.parent.e)

    def test_second_level_dict(self):
        grand = AllowGrand(self.initial, parent={'child.e': 4.0})
        update = {'parent.child.e': 5.0}
        updated = grand(update)
        self.assertTrue(hasattr(updated.parent.child, 'e'))
        self.assertIsInstance(updated.parent.child.e, float)
        self.assertEqual(5.0, updated.parent.child.e)

    def test_second_level_kwarg(self):
        grand = AllowGrand(self.initial, parent={'child.e': 4.0})
        update = {'child.e': 5.0}
        updated = grand(parent=update)
        self.assertTrue(hasattr(updated.parent.child, 'e'))
        self.assertIsInstance(updated.parent.child.e, float)
        self.assertEqual(5.0, updated.parent.child.e)

    def test_second_level_str(self):
        grand = AllowGrand(self.initial, parent={'child.e': 4.0})
        update = '{"parent.child.e": 5.0}'
        updated = grand(update)
        self.assertTrue(hasattr(updated.parent.child, 'e'))
        self.assertIsInstance(updated.parent.child.e, float)
        self.assertEqual(5.0, updated.parent.child.e)


if __name__ == '__main__':
    unittest.main()

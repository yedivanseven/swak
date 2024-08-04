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


class IgnoreChild(Child, ignore_extra=True, raise_extra=False):
    pass


class IgnoreParent(Parent, ignore_extra=True, raise_extra=False):
    child: IgnoreChild


class IgnoreGrand(Grand):
    parent: IgnoreParent


class AllowChild(Child, ignore_extra=False, raise_extra=False):
    pass


class AllowParent(Parent, ignore_extra=False, raise_extra=False):
    child: AllowChild


class AllowGrand(Grand):
    parent: AllowParent


class TestDictOnly(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_fully_nested(self):
        grand = Grand(self.expected)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_first_level_nested_flat_leave(self):
        d = {
            'a': 1,
            'parent.b': 2,
            'parent': {
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_first_level_nested_nested_leave(self):
        d = {
            'a': 1,
            'parent': {
                'b': 2
            },
            'parent.child': {
                'c': 3,
                'd': 'foo'
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_first_level_nested_all_leaves(self):
        d = {
            'a': 1,
            'parent.b': 2,
            'parent.child': {
                'c': 3,
                'd': 'foo'
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_all_levels_nested_one_leave(self):
        d = {
            'a': 1,
            'parent.b': 2,
            'parent.child': {
                'd': 'foo'
            },
            'parent.child.c': 3
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_all_levels_nested_all_leaves(self):
        d = {
            'a': 1,
            'parent.b': 2,
            'parent.child.d': 'foo',
            'parent.child.c': 3
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestDictAndKwargDict(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_fully_nested(self):
        d = {
            'b': 2,
            'child': {
                'c': 3,
                'd': 'foo'
            }
        }
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_partially_nested(self):
        d = {
            'b': 2,
            'child.c': 3,
            'child': {
                'd': 'foo'
            }
        }
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_flat(self):
        d = {
            'b': 2,
            'child.c': 3,
            'child.d': 'foo'
        }
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestStrAndKwargDict(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_fully_nested(self):
        d = {
            'b': 2,
            'child': {
                'c': 3,
                'd': 'foo'
            }
        }
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_partially_nested(self):
        d = {
            'b': 2,
            'child.c': 3,
            'child': {
                'd': 'foo'
            }
        }
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_flat(self):
        d = {
            'b': 2,
            'child.c': 3,
            'child.d': 'foo'
        }
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestDictAndKwargStr(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_fully_nested(self):
        d = '{"b": 2, "child": {"c": 3, "d": "foo"}}'
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_partially_nested(self):
        d = '{"b": 2, "child.c": 3, "child": {"d": "foo"}}'
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_flat(self):
        d = '{"b": 2, "child.c": 3, "child.d": "foo"}'
        grand = Grand({'a': 1}, parent=d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestStrAndKwargStr(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_fully_nested(self):
        d = '{"b": 2, "child": {"c": 3, "d": "foo"}}'
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_partially_nested(self):
        d = '{"b": 2, "child.c": 3, "child": {"d": "foo"}}'
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_flat(self):
        d = '{"b": 2, "child.c": 3, "child.d": "foo"}'
        grand = Grand('{"a": 1}', parent=d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestConflictsPriorities(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_first_level_priority_last_wins(self):
        d = {
            'a': 1,
            'parent.b': 4,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)
        d = {
            'a': 1,
            'parent': {
                'b': 4,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            },
            'parent.b': 2
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_second_level_priority_last_wins(self):
        d = {
            'a': 1,
            'parent': {
                'b': 2,
                'child.c': 4,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)
        d = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 4,
                    'd': 'foo'
                },
                'child.c': 3
            },
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)

    def test_priority_last_wins(self):
        d = {
            'a': 1,
            'parent.b': 2,
            'parent.child.c': 4,
            'parent': {
                'child.c': 5,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)
        d = {
            'a': 1,
            'parent.b': 2,
            'parent': {
                'child.c': 4,
                'child': {
                    'c': 5,
                    'd': 'foo'
                }
            },
            'parent.child.c': 3
        }
        grand = Grand(d)
        self.assertDictEqual(self.expected, grand.as_json)


class TestRaiseExtraFields(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_dict_first_level(self):
        self.expected['parent.e'] = 'bar'
        with self.assertRaises(ExceptionGroup):
            _ = Grand(self.expected)

    def test_dict_second_level(self):
        self.expected['parent']['child.e'] = 'bar'
        with self.assertRaises(ExceptionGroup):
            _ = Grand(self.expected)

    def test_dict_all_levels(self):
        self.expected['parent.child.e'] = 'bar'
        with self.assertRaises(ExceptionGroup):
            _ = Grand(self.expected)

    def test_kwargs_second_level_dict(self):
        with self.assertRaises(ExceptionGroup):
            _ = Grand(self.expected, parent={'child.e': 'bar'})

    def test_kwargs_second_level_str(self):
        with self.assertRaises(ExceptionGroup):
            _ = Grand(self.expected, parent='{"child.e": "bar"}')


class TestIgnoreExtraFields(unittest.TestCase):

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

    def test_dict_first_level(self):
        self.initial['parent.e'] = 'bar'
        grand = IgnoreGrand(self.initial)
        self.assertFalse(hasattr(grand.parent, 'e'))

    def test_dict_second_levels(self):
        self.initial['parent']['child.e'] = 'bar'
        grand = IgnoreGrand(self.initial)
        self.assertFalse(hasattr(grand.parent.child, 'e'))

    def test_dict_all_levels(self):
        self.initial['parent.child.e'] = 'bar'
        grand = IgnoreGrand(self.initial)
        self.assertFalse(hasattr(grand.parent.child, 'e'))

    def test_kwargs_second_level(self):
        grand = IgnoreGrand(self.initial, parent={'child.e': 'bar'})
        self.assertFalse(hasattr(grand.parent.child, 'e'))


class TestAllowExtraFields(unittest.TestCase):

    def setUp(self) -> None:
        self.expected = {
            'a': 1,
            'parent': {
                'b': 2,
                'child': {
                    'c': 3,
                    'd': 'foo'
                }
            }
        }

    def test_dict_first_level(self):
        self.expected['parent.e'] = 'bar'
        grand = AllowGrand(self.expected)
        self.assertTrue(hasattr(grand.parent, 'e'))
        self.assertIsInstance(grand.parent.e, str)
        self.assertEqual('bar', grand.parent.e)

    def test_dict_second_levels(self):
        self.expected['parent']['child.e'] = 'bar'
        grand = AllowGrand(self.expected)
        self.assertTrue(hasattr(grand.parent.child, 'e'))
        self.assertIsInstance(grand.parent.child.e, str)
        self.assertEqual('bar', grand.parent.child.e)

    def test_dict_all_levels(self):
        self.expected['parent.child.e'] = 'bar'
        grand = AllowGrand(self.expected)
        self.assertTrue(hasattr(grand.parent.child, 'e'))
        self.assertIsInstance(grand.parent.child.e, str)
        self.assertEqual('bar', grand.parent.child.e)

    def test_kwargs_second_level(self):
        grand = AllowGrand(self.expected, parent={'child.e': 'bar'})
        self.assertTrue(hasattr(grand.parent.child, 'e'))
        self.assertIsInstance(grand.parent.child.e, str)
        self.assertEqual('bar', grand.parent.child.e)


if __name__ == '__main__':
    unittest.main()

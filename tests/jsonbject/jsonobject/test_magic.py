import unittest
from swak.jsonobject import JsonObject


class Custom:

    def __init__(self, *_):
        pass

    def __repr__(self):
        return 'custom'

    def __eq__(self, _):
        return True


class Simple(JsonObject):
    a: int = 1
    b: str = 'foo'
    c: Custom = True

    d = 2.0

    @property
    def p(self):
        return 42

    def m(self):
        pass

    @classmethod
    def clsm(cls):
        pass

    @staticmethod
    def sm():
        pass


class Other(JsonObject):
    a: int = 2
    b: str = 'bar'


class Extra(JsonObject, ignore_extra=False, raise_extra=False):
    pass


class Empty(JsonObject):
    pass


class TestMagic(unittest.TestCase):

    def test_getitem_field(self):
        simple = Simple()
        a = simple['a']
        self.assertIsInstance(a, int)
        self.assertEqual(1, a)

    def test_getitem_class_variable(self):
        simple = Simple()
        d = simple['d']
        self.assertIsInstance(d, float)
        self.assertEqual(2.0, d)

    def test_getitem_property(self):
        simple = Simple()
        p = simple['p']
        self.assertIsInstance(p, int)
        self.assertEqual(42, p)

    def test_getitem_method(self):
        simple = Simple()
        _ = simple['m']

    def test_getitem_classmethod(self):
        simple = Simple()
        _ = simple['clsm']

    def test_getitem_staticmethod(self):
        simple = Simple()
        _ = simple['clsm']

    def test_getitem_raises_AttributeError(self):
        simple = Simple()
        with self.assertRaises(AttributeError):
            _ = simple['missing']

    def test_getitem_raises_KeyError(self):
        simple = Simple()
        with self.assertRaises(KeyError):
            _ = simple[1]

    def test_iter(self):
        simple = Simple()
        keys = {key for key in simple}
        self.assertSetEqual({'a', 'b', 'c'}, keys)

    def test_str(self):
        simple = Simple()
        actual = str(simple)
        self.assertEqual('{"a": 1, "b": "foo", "c": "custom"}', actual)

    def test_repr(self):
        simple = Simple()
        actual = repr(simple)
        expected = '{\n    "a": 1,\n    "b": "foo",\n    "c": "custom"\n}'
        self.assertEqual(expected, actual)

    def test_equality_on_other_type_is_false(self):
        simple = Simple()
        self.assertFalse(simple == 2)

    def test_equality_on_other_jsonobject_is_false(self):
        simple = Simple()
        other = Other()
        self.assertFalse(simple == other)

    def test_equality_on_other_content_is_false(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'bar'})
        self.assertFalse(simple == other)

    def test_equality_on_same_content(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'foo'})
        self.assertEqual(simple, other)

    def test_equality_on_self(self):
        simple = Simple()
        self.assertEqual(simple, simple)

    def test_inequality_on_other_type(self):
        simple = Simple()
        self.assertNotEqual(simple, 2)

    def test_inequality_on_other_jsonobject(self):
        simple = Simple()
        other = Other()
        self.assertNotEqual(simple, other)

    def test_inequality_on_other_content(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'bar'})
        self.assertNotEqual(simple, other)

    def test_inequality_on_same_content_is_false(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'foo'})
        self.assertFalse(simple != other)

    def test_inequality_on_self_is_false(self):
        simple = Simple()
        self.assertFalse(simple != simple)

    def test_contains_true(self):
        simple = Simple()
        self.assertIn('a', simple)
        self.assertIn('b', simple)
        self.assertIn('c', simple)

    def test_contains_false(self):
        simple = Simple()
        self.assertNotIn('d', simple)
        self.assertNotIn(3.0, simple)
        self.assertNotIn(True, simple)

    def test_len(self):
        simple = Simple()
        self.assertEqual(3, len(simple))

    def test_len_zero(self):
        empty = Empty()
        self.assertEqual(0, len(empty))

    def test_bool_true(self):
        simple = Simple()
        self.assertTrue(simple)

    def test_bool_emtpy_false(self):
        empty = Empty()
        self.assertFalse(empty)

    def test_dictionary_unwrapping(self):
        simple = Simple()
        actual = {**simple}
        expected = {'a': 1, 'b': 'foo', 'c': simple.c}
        self.assertDictEqual(expected, actual)

    def test_dictionary_unwrapping_empty(self):
        empty = Empty()
        self.assertDictEqual({}, {**empty})

    def test_dict_conversion(self):
        simple = Simple()
        actual = dict(simple)
        expected = {'a': 1, 'b': 'foo', 'c': simple.c}
        self.assertDictEqual(expected, actual)

    def test_dict_conversion_empty(self):
        empty = Empty()
        self.assertDictEqual({}, dict(empty))


if __name__ == '__main__':
    unittest.main()

import unittest
from swak.jsonobject import JsonObject


class Simple(JsonObject):
    a: int = 1
    b: str = 'foo'


class Other(JsonObject):
    a: int = 2
    b: str = 'bar'


class Dummy:

    def __init__(self, *args):
        pass


class Custom(JsonObject):
    d: Dummy = 1


class Empty(JsonObject):
    pass


class TestMagic(unittest.TestCase):

    def test_getitem(self):
        simple = Simple()
        a = simple['a']
        b = simple['b']
        self.assertIsInstance(a, int)
        self.assertEqual(1, a)
        self.assertIsInstance(b, str)
        self.assertEqual('foo', b)

    def test_getitem_raises_KeyError(self):
        simple = Simple()
        with self.assertRaises(AttributeError):
            _ = simple['c']

    def test_getitem_raises_KeyError_on_non_string_Key(self):
        simple = Simple()
        with self.assertRaises(KeyError):
            _ = simple[1]

    def test_iter(self):
        simple = Simple()
        keys = [key for key in simple]
        self.assertListEqual(['a', 'b'], keys)

    def test_str(self):
        simple = Simple()
        actual = str(simple)
        self.assertEqual('{"a": 1, "b": "foo"}', actual)

    def test_str_custom(self):
        custom = Custom()
        actual = str(custom)
        self.assertEqual(f'{{"d": "{custom.d}"}}', actual)

    def test_repr(self):
        simple = Simple()
        actual = repr(simple)
        expected = '{\n    "a": 1,\n    "b": "foo"\n}'
        self.assertEqual(expected, actual)

    def test_repr_custom(self):
        custom = Custom()
        actual = repr(custom)
        expected = f'{{\n    "d": "{str(custom.d)}"\n}}'
        self.assertEqual(expected, actual)

    def test_equality_on_other_type_is_false(self):
        simple = Simple()
        self.assertNotEqual(simple, 2)

    def test_equality_on_other_jsonobject_is_false(self):
        simple = Simple()
        other = Other()
        self.assertNotEqual(simple, other)

    def test_equality_on_other_content_is_false(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'bar'})
        self.assertNotEqual(simple, other)

    def test_equality_on_same_content_is_true(self):
        simple = Simple()
        other = Simple({'a': 1, 'b': 'foo'})
        self.assertEqual(simple, other)

    def test_equality_on_self_is_true(self):
        simple = Simple()
        self.assertEqual(simple, simple)

    def test_contains_true(self):
        simple = Simple()
        self.assertIn('a', simple)
        self.assertIn('b', simple)

    def test_contains_false(self):
        simple = Simple()
        self.assertNotIn('c', simple)
        self.assertNotIn(3.0, simple)

    def test_len(self):
        simple = Simple()
        self.assertEqual(2, len(simple))

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
        expected = {'a': 1, 'b': 'foo'}
        self.assertDictEqual(expected, actual)

    def test_dictionary_unwrapping_empty(self):
        empty = Empty()
        self.assertDictEqual({}, {**empty})

    def test_dict_conversion(self):
        simple = Simple()
        actual = dict(simple)
        expected = {'a': 1, 'b': 'foo'}
        self.assertDictEqual(expected, actual)

    def test_dict_conversion_empty(self):
        empty = Empty()
        self.assertDictEqual({}, dict(empty))


if __name__ == '__main__':
    unittest.main()

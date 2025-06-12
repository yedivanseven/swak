import unittest
from unittest.mock import patch
from pandas import Series
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ParseError, ValidationErrors


class Child(JsonObject):
    b: float = 2.0
    c: int = 1
    d: bool = True


class Parent(JsonObject):
    a: str = 'foo'
    child: Child = Child()


class Respect(JsonObject, respect_none=True):
    a: Maybe[int](int) = 42
    b: str = 'foo'


class TestEmpty(unittest.TestCase):

    def setUp(self):
        self.flat = Child()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Child)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, float)
        self.assertEqual(2.0, obj.b)
        self.assertTrue(hasattr(obj, 'c'))
        self.assertIsInstance(obj.c, int)
        self.assertEqual(1, obj.c)
        self.assertTrue(hasattr(obj, 'd'))
        self.assertIsInstance(obj.d, bool)
        self.assertTrue(obj.d)

    def test_no_arg(self):
        updated = self.flat()
        self.check_attributes(updated)

    def test_empty_dict(self):
        updated = self.flat({})
        self.check_attributes(updated)

    def test_none(self):
        updated = self.flat(None)
        self.check_attributes(updated)

    def test_empty_series(self):
        updated = self.flat(Series())
        self.check_attributes(updated)

    def test_str_of_empty_dict(self):
        updated = self.flat('{}')
        self.check_attributes(updated)

    def test_bytes_of_empty_dict(self):
        updated = self.flat(b'{}')
        self.check_attributes(updated)

    def test_bytearray_of_empty_dict(self):
        updated = self.flat(bytearray(b'{}'))
        self.check_attributes(updated)


class TestNone(unittest.TestCase):

    def setUp(self):
        self.flat = Child()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Child)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, float)
        self.assertEqual(2.0, obj.b)
        self.assertTrue(hasattr(obj, 'c'))
        self.assertIsInstance(obj.c, int)
        self.assertEqual(1, obj.c)
        self.assertTrue(hasattr(obj, 'd'))
        self.assertIsInstance(obj.d, bool)
        self.assertTrue(obj.d)
        self.assertFalse(hasattr(obj, 'bar'))

    def test_dict_none(self):
        updated = self.flat({'bar': None})
        self.check_attributes(updated)

    def test_series_none(self):
        updated = self.flat(Series({'bar': None}))
        self.check_attributes(updated)

    def test_str_of_dict_none(self):
        updated = self.flat("{'bar': None}")
        self.check_attributes(updated)

    def test_json_of_dict_null(self):
        updated = self.flat('{"bar": null}')
        self.check_attributes(updated)

    def test_bytes_of_dict_null(self):
        updated = self.flat(b'{"bar": null}')
        self.check_attributes(updated)

    def test_bytearray_of_dict_null(self):
        updated = self.flat(bytearray(b'{"bar": null}'))
        self.check_attributes(updated)

    def test_kwarg_none(self):
        updated = self.flat(bar=None)
        self.check_attributes(updated)


class TestFlat(unittest.TestCase):

    def setUp(self):
        self.flat = Child()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Child)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, float)
        self.assertEqual(3.0, obj.b)
        self.assertTrue(hasattr(obj, 'c'))
        self.assertIsInstance(obj.c, int)
        self.assertEqual(42, obj.c)
        self.assertTrue(hasattr(obj, 'd'))
        self.assertIsInstance(obj.d, bool)
        self.assertTrue(obj.d)
        self.assertFalse(hasattr(obj, 'bar'))

    def test_dict(self):
        updated = self.flat({'b': 3.0, 'c': 42})
        self.check_attributes(updated)

    def test_dict_and_kwarg(self):
        updated = self.flat({'b': 3.0}, c=42)
        self.check_attributes(updated)

    def test_kwargs(self):
        updated = self.flat(b=3.0, c=42)
        self.check_attributes(updated)

    def test_empty_dict_and_kwargs(self):
        updated = self.flat({}, b=3.0, c=42)
        self.check_attributes(updated)

    def test_empty_series_and_kwargs(self):
        updated = self.flat(Series(), b=3.0, c=42)
        self.check_attributes(updated)

    def test_none_and_kwargs(self):
        updated = self.flat(None, b=3.0, c=42)
        self.check_attributes(updated)

    def test_str_of_empty_dict_and_kwargs(self):
        updated = self.flat('{}', b=3.0, c=42)
        self.check_attributes(updated)

    def test_bytes_of_empty_dict_and_kwargs(self):
        updated = self.flat(b'{}', b=3.0, c=42)
        self.check_attributes(updated)

    def test_bytearray_of_empty_dict_and_kwargs(self):
        updated = self.flat(bytearray(b'{}'), b=3.0, c=42)
        self.check_attributes(updated)

    def test_series(self):
        updated = self.flat(Series({'b': 3.0, 'c': 42}))
        self.check_attributes(updated)

    def test_series_and_kwarg(self):
        updated = self.flat(Series({'b': 3.0}), c=42)
        self.check_attributes(updated)

    def test_str(self):
        updated = self.flat("{'b': 3.0, 'c': 42}")
        self.check_attributes(updated)

    def test_str_and_kwarg(self):
        updated = self.flat("{'b': 3.0}", c=42)
        self.check_attributes(updated)

    def test_json(self):
        updated = self.flat('{"b": 3.0, "c": 42}')
        self.check_attributes(updated)

    def test_json_and_kwarg(self):
        updated = self.flat('{"b": 3.0}', c=42)
        self.check_attributes(updated)

    def test_bytes(self):
        updated = self.flat(b'{"b": 3.0, "c": 42}')
        self.check_attributes(updated)

    def test_bytes_and_kwarg(self):
        updated = self.flat(b'{"b": 3.0}', c=42)
        self.check_attributes(updated)

    def test_bytearray(self):
        updated = self.flat(bytearray(b'{"b": 3.0, "c": 42}'))
        self.check_attributes(updated)

    def test_bytearray_and_kwarg(self):
        updated = self.flat(bytearray(b'{"b": 3.0}'), c=42)
        self.check_attributes(updated)

    def test_self(self):
        update = Child(b=3.0, c=42)
        updated = self.flat(update)
        self.check_attributes(updated)

    def test_kwarg_trumps_dict(self):
        updated = self.flat({'b': 3.0, 'c': 4}, c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_series(self):
        updated = self.flat(Series({'b': 3.0, 'c': 4}), c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_str(self):
        updated = self.flat("{'b': 3.0, 'c': 4}", c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_json(self):
        updated = self.flat('{"b": 3.0, "c": 4}', c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_bytes(self):
        updated = self.flat(b'{"b": 3.0, "c": 4}', c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_bytearray_and_kwarg(self):
        updated = self.flat(bytearray(b'{"b": 3.0, "c": 4}'), c=42)
        self.check_attributes(updated)

    def test_kwarg_trumps_self(self):
        update = Child(b=3.0, c=4)
        updated = self.flat(update, c=42)
        self.check_attributes(updated)

    def test_kwarg_none_ignored(self):
        updated = self.flat({'b': 3.0, 'c': 42}, c=None)
        self.check_attributes(updated)

    def test_extra_kwarg_none_ignored(self):
        updated = self.flat({'b': 3.0, 'c': 42}, bar=None)
        self.check_attributes(updated)

    def test_dict_and_kwarg_collapse(self):
        updated = self.flat({'b': 3.0, 'c': {42: 'answer'}}, c=42)
        self.check_attributes(updated)


class TestExtra(unittest.TestCase):

    class TrueTrue(JsonObject, ignore_extra=True, raise_extra=True):
        pass

    class TrueFalse(JsonObject, ignore_extra=True, raise_extra=False):
        pass

    class FalseTrue(JsonObject, ignore_extra=False, raise_extra=True):
        pass

    class FalseFalse(JsonObject, ignore_extra=False, raise_extra=False):
        pass

    def test_true_true_ignore_empty(self):
        extra = self.TrueTrue()
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_true_true_ignore_non_empty(self):
        extra = self.TrueTrue(a=1)
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_true_false_ignore_empty(self):
        extra = self.TrueFalse()
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_true_false_ignore_non_empty(self):
        extra = self.TrueFalse(a=1)
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_false_true_ignore_empty(self):
        extra = self.FalseTrue()
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_false_false_ignore_empty(self):
        extra = self.FalseFalse()
        updated = extra(a=42, b='foo')
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_false_false_accept_non_empty(self):
        extra = self.FalseFalse(a=1)
        updated = extra(a=42, b='foo')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(42, updated.a)
        self.assertFalse(hasattr(updated, 'b'))

    def test_false_false_nest_non_empty(self):
        extra = self.FalseFalse(a=1)
        updated = extra(a={42: 'answer'}, b='foo')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertDictEqual({42: 'answer'}, updated.a)
        self.assertFalse(hasattr(updated, 'b'))

    def test_false_false_collapse_non_empty(self):
        extra = self.FalseFalse(a={42: 'answer'})
        updated = extra(a=42, b='foo')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(42, updated.a)
        self.assertFalse(hasattr(updated, 'b'))


class TestRespect(unittest.TestCase):

    def setUp(self):
        self.respect = Respect()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Respect)
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsNone(obj.a)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertEqual('foo', obj.b)

    def test_dict(self):
        updated = self.respect({'a': None})
        self.check_attributes(updated)

    def test_none_str_dict(self):
        updated = self.respect({'a': 'None'})
        self.check_attributes(updated)

    def test_null_str_dict(self):
        updated = self.respect({'a': 'null'})
        self.check_attributes(updated)

    def test_kwarg(self):
        updated = self.respect(a=None)
        self.check_attributes(updated)

    def test_none_str_kwarg(self):
        updated = self.respect(a='None')
        self.check_attributes(updated)

    def test_null_str_kwarg(self):
        updated = self.respect(a='None')
        self.check_attributes(updated)

    def test_series(self):
        updated = self.respect(Series({'a': None}))
        self.check_attributes(updated)

    def test_none_str_series(self):
        updated = self.respect(Series({'a': 'None'}))
        self.check_attributes(updated)

    def test_null_str_series(self):
        updated = self.respect(Series({'a': 'null'}))
        self.check_attributes(updated)

    def test_str(self):
        updated = self.respect("{'a': None}")
        self.check_attributes(updated)

    def test_none_str_str(self):
        updated = self.respect("{'a': 'None'}")
        self.check_attributes(updated)

    def test_null_str_str(self):
        updated = self.respect("{'a': 'null'}")
        self.check_attributes(updated)

    def test_json(self):
        updated = self.respect('{"a": null}')
        self.check_attributes(updated)

    def test_bytes_of_dict_null(self):
        updated = self.respect(b'{"a": null}')
        self.check_attributes(updated)

    def test_bytearray_of_dict_null(self):
        updated = self.respect(bytearray(b'{"a": null}'))
        self.check_attributes(updated)

    def test_self(self):
        updated = self.respect(Respect(a=None))
        self.check_attributes(updated)

    def test_raises_on_maybe_not(self):
        with self.assertRaises(ValidationErrors):
            _ = Respect(b=None)


class TestExtraRespect(unittest.TestCase):

    class ExtraRespect(
        JsonObject,
        ignore_extra=False,
        raise_extra=False,
        respect_none=True
    ):
        pass

    def test_ignore_empty(self):
        extra_respect = self.ExtraRespect()
        updated = extra_respect(a=None, b=42)
        self.assertFalse(hasattr(updated, 'a'))
        self.assertFalse(hasattr(updated, 'b'))

    def test_accept_non_empty(self):
        extra_respect = self.ExtraRespect(a=1)
        updated = extra_respect(a=None, b=42)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsNone(updated.a)
        self.assertFalse(hasattr(updated, 'b'))


class TestTypeCasting(unittest.TestCase):

    def setUp(self):
        self.flat = Child()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Child)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, float)
        self.assertEqual(3.0, obj.b)
        self.assertTrue(hasattr(obj, 'c'))
        self.assertIsInstance(obj.c, int)
        self.assertEqual(42, obj.c)
        self.assertTrue(hasattr(obj, 'd'))
        self.assertIsInstance(obj.d, bool)
        self.assertTrue(obj.d)

    def test_dict(self):
        updated = self.flat({'b': 3, 'c': 42.0, 'd': 'foo'})
        self.check_attributes(updated)

    def test_dict_and_kwarg(self):
        updated = self.flat({'b': 3, 'c': 42.0}, d='foo')
        self.check_attributes(updated)

    def test_kwargs(self):
        updated = self.flat(b=3, c=42.0, d='foo')
        self.check_attributes(updated)

    def test_series(self):
        updated = self.flat(Series({'b': 3, 'c': 42.0, 'd': 'foo'}))
        self.check_attributes(updated)

    def test_series_and_kwarg(self):
        updated = self.flat(Series({'b': 3, 'c': 42.0}), d='foo')
        self.check_attributes(updated)

    def test_str(self):
        updated = self.flat("{'b': 3, 'c': 42.0, 'd': 'foo'}")
        self.check_attributes(updated)

    def test_str_and_kwarg(self):
        updated = self.flat("{'b': 3, 'c': 42}", d='foo')
        self.check_attributes(updated)

    def test_json(self):
        updated = self.flat('{"b": 3, "c": 42.0, "d": "foo"}')
        self.check_attributes(updated)

    def test_json_and_kwarg(self):
        updated = self.flat('{"b": 3, "c": 42.0}', d='foo')
        self.check_attributes(updated)

    def test_bytes(self):
        updated = self.flat(b'{"b": 3, "c": 42.0, "d": "foo"}')
        self.check_attributes(updated)

    def test_bytes_and_kwarg(self):
        updated = self.flat(b'{"b": 3, "c": 42.0}', d='foo')
        self.check_attributes(updated)

    def test_bytearray(self):
        updated = self.flat(bytearray(b'{"b": 3, "c": 42.0, "d": "foo"}'))
        self.check_attributes(updated)

    def test_bytearray_and_kwarg(self):
        updated = self.flat(bytearray(b'{"b": 3, "c": 42.0}'), d='foo')
        self.check_attributes(updated)


class TestExceptions(unittest.TestCase):

    def setUp(self):
        self.flat = Child()

    def test_uncastable_str(self):
        expected = "Could not parse hello world as JSON!"
        with self.assertRaises(ParseError) as error:
            _ = self.flat('hello world')
        self.assertEqual(expected, str(error.exception))

    def test_uncastable_bytes(self):
        expected = "Could not parse b'hello world' as JSON!"
        with self.assertRaises(ParseError) as error:
            _ = self.flat(b'hello world')
        self.assertEqual(expected, str(error.exception))

    def test_dict(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat({'b': 'hello world'})

    def test_dict_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat({'b': 'hello'}, c='world')

    def test_kwargs(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(b='hello', c='world')

    def test_series(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(Series({'b': 'hello world'}))

    def test_series_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(Series({'b': 'hello'}), c='world')

    def test_str(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat("{'b': 'hello world'}")

    def test_str_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat("{'b': 'hello'}", c='world')

    def test_json(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat('{"b": "hello world"}')

    def test_json_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat('{"b": "hello"}', c='world')

    def test_bytes(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(b'{"b": "hello world"}')

    def test_bytes_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(b'{"b": "hello"}', c='world')

    def test_bytearray(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(bytearray(b'{"b": "hello world"}'))

    def test_bytearray_and_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = self.flat(bytearray(b'{"b": "hello"}'), c='world')


class TestNesting(unittest.TestCase):

    def setUp(self):
        self.nested = Parent()

    def check_attributes(self, obj):
        self.assertIsInstance(obj, Parent)
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(2.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(42, obj.child.c)
        self.assertTrue(hasattr(obj.child, 'd'))
        self.assertIsInstance(obj.child.d, bool)
        self.assertTrue(obj.child.d)

    def test_dict_dict(self):
        updated = self.nested({'child': {'c': 42}})
        self.check_attributes(updated)

    def test_nest(self):
        updated = self.nested({'child': 42}, child={'c': 42})
        self.check_attributes(updated)

    def test_collapse(self):
        updated = self.nested(
            {'child': {'c': {42: 'answer'}}},
            child={'c': 42}
        )
        self.check_attributes(updated)

    def test_dict_self(self):
        updated = self.nested({'child': Child({'c': 42})})
        self.check_attributes(updated)

    def test_dict_series(self):
        updated = self.nested({'child': Series({'c': 42})})
        self.check_attributes(updated)

    def test_dict_str(self):
        updated = self.nested({'child': "{'c': 42}"})
        self.check_attributes(updated)

    def test_dict_json(self):
        updated = self.nested({'child': '{"c": 42}'})
        self.check_attributes(updated)

    def test_dict_bytes(self):
        updated = self.nested({'child': b'{"c": 42}'})
        self.check_attributes(updated)

    def test_dict_bytearray(self):
        updated = self.nested({'child': bytearray(b'{"c": 42}')})
        self.check_attributes(updated)

    def test_kwarg_dict(self):
        updated = self.nested(child={'c': 42})
        self.check_attributes(updated)

    def test_kwarg_self(self):
        updated = self.nested(child=Child({'c': 42}))
        self.check_attributes(updated)

    def test_kwarg_series(self):
        updated = self.nested(child=Series({'c': 42}))
        self.check_attributes(updated)

    def test_kwarg_str(self):
        updated = self.nested(child="{'c': 42}")
        self.check_attributes(updated)

    def test_kwarg_json(self):
        updated = self.nested(child='{"c": 42}')
        self.check_attributes(updated)

    def test_kwarg_bytes(self):
        updated = self.nested(child=b'{"c": 42}')
        self.check_attributes(updated)

    def test_kwarg_bytearray(self):
        updated = self.nested(child=bytearray(b'{"c": 42}'))
        self.check_attributes(updated)

    def test_series_dict(self):
        updated = self.nested(Series({'child': {'c': 42}}))
        self.check_attributes(updated)

    def test_series_self(self):
        updated = self.nested(Series({'child': Child({'c': 42})}))
        self.check_attributes(updated)

    def test_series_series(self):
        updated = self.nested(Series({'child': Series({'c': 42})}))
        self.check_attributes(updated)

    def test_series_str(self):
        updated = self.nested(Series({'child': "{'c': 42}"}))
        self.check_attributes(updated)

    def test_series_json(self):
        updated = self.nested(Series({'child': '{"c": 42}'}))
        self.check_attributes(updated)

    def test_series_bytes(self):
        updated = self.nested(Series({'child': b'{"c": 42}'}))
        self.check_attributes(updated)

    def test_series_bytearray(self):
        updated = self.nested(Series({'child': bytearray(b'{"c": 42}')}))
        self.check_attributes(updated)

    def test_str(self):
        updated = self.nested("{'child': {'c': 42}}")
        self.check_attributes(updated)

    def test_json(self):
        updated = self.nested('{"child": {"c": 42}}')
        self.check_attributes(updated)

    def test_bytes(self):
        updated = self.nested(b'{"child": {"c": 42}}')
        self.check_attributes(updated)

    def test_bytearray(self):
        updated = self.nested(bytearray(b'{"child": {"c": 42}}'))
        self.check_attributes(updated)


class TestMagic(unittest.TestCase):

    def test_or_calls_update(self):
        child = Child()
        with patch.object(child, '__call__') as mock:
            _ = child | 'foo'
            mock.assert_called_once_with('foo')


if __name__ == '__main__':
    unittest.main()

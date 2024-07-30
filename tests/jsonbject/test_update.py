import unittest
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ParseError, CastError


class Flat(JsonObject):
    a: int


class FlatDef(JsonObject):
    a: int = 1


class Nested(JsonObject):
    b: Flat


class NestedDef(JsonObject):
    b: FlatDef = FlatDef()


class TestFlatEmpty(unittest.TestCase):

    def test_empty(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat()
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)

    def test_none(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat(None)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)

    def test_dict(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({})
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)

    def test_default_empty(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat()
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)

    def test_default_none(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat(None)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)

    def test_default_dict(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat({})
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(1, updated.a)


class TestNestedEmpty(unittest.TestCase):

    def test_empty(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested()
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(1, updated.b.a)

    def test_none(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested(b=None)
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(1, updated.b.a)

    def test_default_empty(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested()
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(1, updated.b.a)

    def test_default_none(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested(b=None)
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(1, updated.b.a)

    def test_default_dict(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested(b={})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(1, updated.b.a)


class TestFlat(unittest.TestCase):

    def test_dict(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({'a': 2})
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_kwargs(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat(a=2)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_str(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat('{"a": 2}')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_kwarg_trumps_dict(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({'a': 3}, a=2)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_kwarg_trumps_str(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat('{"a": 3}', a=2)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_none_kwarg_ignored(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({'a': 2}, a=None)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_extra(self):
        flat = Flat(a=1)
        updated = flat({'b': 1})
        self.assertFalse(hasattr(updated, 'b'))

    def test_extra_none_kwarg_ignored(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({'a': 2}, b=None)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_extra_kwarg_ignored(self):
        flat = Flat(a=1)
        self.assertTrue(callable(flat))
        updated = flat({'a': 2}, b='foo')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_parse_error(self):
        flat = Flat(a=1)
        with self.assertRaises(ParseError):
            _ = flat('foo')

    def test_cast_error(self):
        flat = Flat(a=1)
        with self.assertRaises(CastError):
            _ = flat(a='foo')

    def test_default_dict(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat({'a': 2})
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_default_kwargs(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat(a=2)
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_default_str(self):
        flat = FlatDef()
        self.assertTrue(callable(flat))
        updated = flat('{"a": 2}')
        self.assertTrue(hasattr(updated, 'a'))
        self.assertIsInstance(updated.a, int)
        self.assertEqual(2, updated.a)

    def test_default_parse_error(self):
        flat = FlatDef()
        with self.assertRaises(ParseError):
            _ = flat('foo')

    def test_default_cast_error(self):
        flat = FlatDef()
        with self.assertRaises(CastError):
            _ = flat(a='foo')


class TestNested(unittest.TestCase):

    def test_dict(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 2}})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_kwargs(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested(b={'a': 2})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_str(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested('{"b": {"a": 2}}')
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_dict_kwarg_trumps_dict(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 3}}, b={'a': 2})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_dict_kwarg_trumps_str(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested('{"b": {"a": 3}}', b={'a': 2})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_extra(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 2}, 'c': 'foo'})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_none_kwarg_ignored(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 2}}, b=None)
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_extra_none_kwarg_ignored(self):
        nested = Nested(b={'a': 1})
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 2}}, c=None)
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_cast_error_dict(self):
        nested = Nested(b={'a': 1})
        with self.assertRaises(CastError):
            _ = nested({'b': {'a': 'foo'}})

    def test_cast_error_kwargs(self):
        nested = Nested(b={'a': 1})
        with self.assertRaises(CastError):
            _ = nested(b={'a': 'foo'})

    def test_cast_error_str(self):
        nested = Nested(b={'a': 1})
        with self.assertRaises(CastError):
            _ = nested('{"b": {"a": "foo"}}')

    def test_default_dict(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested({'b': {'a': 2}})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_default_kwargs(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested(b={'a': 2})
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_default_str(self):
        nested = NestedDef()
        self.assertTrue(callable(nested))
        updated = nested('{"b": {"a": 2}}')
        self.assertTrue(hasattr(updated.b, 'a'))
        self.assertIsInstance(updated.b.a, int)
        self.assertEqual(2, updated.b.a)

    def test_default_cast_error_dict(self):
        nested = NestedDef()
        with self.assertRaises(CastError):
            _ = nested({'b': {'a': 'foo'}})

    def test_default_cast_error_kwargs(self):
        nested = NestedDef()
        with self.assertRaises(CastError):
            _ = nested(b={'a': 'foo'})

    def test_default_cast_error_str(self):
        nested = NestedDef()
        with self.assertRaises(CastError):
            _ = nested('{"b": {"a": "foo"}}')


class TestAllowExtraFields(unittest.TestCase):

    def test_flat_dict(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False
        ):
            a: int

        extra = Extra(a=1, b=2)
        updated = extra({'b': 'foo'})
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsInstance(updated.b, str)
        self.assertEqual(updated.b, 'foo')

    def test_flat_kwargs(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False
        ):
            a: int

        extra = Extra(a=1, b=2)
        updated = extra(b='foo')
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsInstance(updated.b, str)
        self.assertEqual(updated.b, 'foo')

    def test_nested_dict(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False
        ):
            a: int

        class DeepExtra(JsonObject):
            b: Extra

        extra = DeepExtra(b={'a': 1, 'c': 2})
        updated = extra({'b': {'a': 1, 'c': 'foo'}})
        self.assertTrue(hasattr(updated.b, 'c'))
        self.assertIsInstance(updated.b.c, str)
        self.assertEqual(updated.b.c, 'foo')

    def test_nested_kwargs(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False
        ):
            a: int

        class DeepExtra(JsonObject):
            b: Extra

        extra = DeepExtra(b={'a': 1, 'c': 2})
        updated = extra(b={'a': 1, 'c': 'foo'})
        self.assertTrue(hasattr(updated.b, 'c'))
        self.assertIsInstance(updated.b.c, str)
        self.assertEqual(updated.b.c, 'foo')

    def test_none_dict(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False,
            respect_none=True
        ):
            a: int

        extra = Extra({'a': 1, 'b': 2})
        updated = extra({'b': None})
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsNone(updated.b)

    def test_none_kwarg(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False,
            respect_none=True
        ):
            a: int

        extra = Extra({'a': 1, 'b': 2})
        updated = extra(b=None)
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsNone(updated.b)

    def test_none_kwarg_trumps_dict(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False,
            respect_none=True
        ):
            a: int

        extra = Extra({'a': 1, 'b': 2})
        updated = extra({'b': 3}, b=None)
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsNone(updated.b)

    def test_none_str(self):

        class Extra(
            JsonObject,
            ignore_extra=False,
            raise_extra=False,
            respect_none=True
        ):
            a: int

        extra = Extra({'a': 1, 'b': 2})
        updated = extra('{"b": null}')
        self.assertTrue(hasattr(updated, 'b'))
        self.assertIsNone(updated.b)


class TestMaybe(unittest.TestCase):

    class NoneDefault(JsonObject):
        a: Maybe[int](int) = None
        b: str = 'foo'

    class MaybeField(JsonObject, respect_none=True):
        a: Maybe[int](int)
        b: str = 'foo'

    def test_none_default_is_overridden_by_dict(self):
        none_default = self.NoneDefault()({'a': 1})
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(1, none_default.a)

    def test_none_default_is_overridden_by_kwarg(self):
        none_default = self.NoneDefault()(a=1)
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(1, none_default.a)

    def test_none_default_is_overridden_by_dict_kwarg(self):
        none_default = self.NoneDefault()({'a': 1}, a=2)
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(2, none_default.a)

    def test_none_dict_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': None})
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_null_str_dict_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': 'null'})
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_str_dict_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': 'None'})
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})(a=None)
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_null_str_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})(a='null')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_str_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})(a='None')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_dict_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': 2}, a=None)
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_null_str_dict_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': 2}, a='null')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_str_dict_kwarg_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})({'a': 2}, a='None')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_str_overwrites_value(self):
        maybe_default = self.MaybeField({'a': 1})('{"a": null}')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)


if __name__ == '__main__':
    unittest.main()

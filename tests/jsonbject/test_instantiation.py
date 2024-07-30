import unittest
from unittest.mock import Mock
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ParseError, CastError


class Simple(JsonObject):
    a: int
    b: str


class Empty(JsonObject):
    pass


class TestInstantiation(unittest.TestCase):

    def test_dict(self):
        simple = Simple({'a': 1, 'b': 'foo'})
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_dict_and_kwarg(self):
        simple = Simple({'b': 'foo'}, a=1)
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_kwargs(self):
        simple = Simple(a=1, b='foo')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_empty_and_kwargs(self):
        simple = Simple({}, a=1, b='foo')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_none_and_kwargs(self):
        simple = Simple(None, a=1, b='foo')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int, type(simple.a))
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_str(self):
        simple = Simple('{"a": 1, "b": "foo"}')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_str_and_kwarg(self):
        simple = Simple('{"b": "foo"}', a=1)
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('foo', simple.b)

    def test_self(self):
        simple = Simple({'a': 1, 'b': 'foo'})
        derived = Simple(simple)
        self.assertTrue(hasattr(derived, 'a'))
        self.assertIsInstance(derived.a, int)
        self.assertEqual(1, derived.a)
        self.assertTrue(hasattr(derived, 'b'))
        self.assertIsInstance(derived.b, str)
        self.assertEqual('foo', derived.b)

    def test_kwarg_trumps_dict(self):
        simple = Simple({'a': 1, 'b': 'foo'}, b='bar')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('bar', simple.b)

    def test_kwarg_trumps_str(self):
        simple = Simple('{"a": 1, "b": "foo"}', b='bar')
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('bar', simple.b)

    def test_kwarg_none_ignored(self):
        simple = Simple({'a': 1, 'b': 'bar'}, b=None)
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('bar', simple.b)

    def test_extra_kwarg_none_ignored(self):
        simple = Simple({'a': 1, 'b': 'bar'}, c=None)
        self.assertTrue(hasattr(simple, 'a'))
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1, simple.a)
        self.assertTrue(hasattr(simple, 'b'))
        self.assertIsInstance(simple.b, str)
        self.assertEqual('bar', simple.b)


class TestNone(unittest.TestCase):

    class MaybeField(JsonObject, respect_none=True):
        a: Maybe[int](int)

    def test_dict(self):
        maybe = self.MaybeField({'a': None})
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_null_str_dict(self):
        maybe = self.MaybeField({'a': 'null'})
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_none_str_dict(self):
        maybe = self.MaybeField({'a': 'None'})
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_kwarg(self):
        maybe = self.MaybeField(a=None)
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_null_str_kwarg(self):
        maybe = self.MaybeField(a='null')
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_none_str_kwarg(self):
        maybe = self.MaybeField(a='None')
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_empty_and_kwarg(self):
        maybe = self.MaybeField({}, a=None)
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_none_and_kwarg(self):
        maybe = self.MaybeField(None, a=None)
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_kwarg_trumps_dict(self):
        maybe = self.MaybeField({'a': 2}, a=None)
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)

    def test_str(self):
        maybe = self.MaybeField('{"a": null}')
        self.assertTrue(hasattr(maybe, 'a'))
        self.assertIsNone(maybe.a)


class TestTypeCasting(unittest.TestCase):

    def test_dict(self):
        simple = Simple({'a': 1.0, 'b': 2.0})
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_dict_and_kwarg(self):
        simple = Simple({'b': 2.0}, a=1.0)
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_kwargs(self):
        simple = Simple(b=2.0, a=1.0)
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_str(self):
        simple = Simple('{"a": 1.0, "b": 2.0}')
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_str_and_kwarg(self):
        simple = Simple('{"b": 2.0}', a=1.0)
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_empty_and_kwarg(self):
        simple = Simple({}, a=1.0, b=2.0)
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_none_and_kwarg(self):
        simple = Simple(None, a=1.0, b=2.0)
        self.assertIsInstance(simple.a, int)
        self.assertEqual(1.0, simple.a)
        self.assertIsInstance(simple.b, str)
        self.assertEqual('2.0', simple.b)

    def test_type_cast_called(self):
        mock = Mock()

        class Custom(JsonObject):
            a: mock

        _ = Custom(a=1)
        mock.assert_called_once()
        mock.assert_called_once_with(1)


class TestEmpty(unittest.TestCase):

    def test_empty_no_arg(self):
        _ = Empty()

    def test_empty_empty_dict(self):
        _ = Empty({})

    def test_empty_none(self):
        _ = Empty(None)


class TestExceptions(unittest.TestCase):

    def test_parse_dict_missing_fields(self):
        with self.assertRaises(ParseError):
            _ = Simple({'b': 'foo'})

    def test_parse_kwargs_missing_fields(self):
        with self.assertRaises(ParseError):
            _ = Simple(a=1)

    def test_parse_str_missing_fields(self):
        with self.assertRaises(ParseError):
            _ = Simple('{"b": "foo"}')

    def test_cast_dict_wrong_field(self):
        with self.assertRaises(CastError):
            _ = Simple({'a': 'foo', 'b': 'bar'})

    def test_cast_dict_and_kwarg_wrong_field(self):
        with self.assertRaises(CastError):
            _ = Simple({'b': 'bar'}, a='foo')

    def test_cast_kwargs_wrong_field(self):
        with self.assertRaises(CastError):
            _ = Simple(b='bar', a='foo')

    def test_cast_str_wrong_field(self):
        with self.assertRaises(CastError):
            _ = Simple('{"a": "foo", "b": "bar"}')

    def test_cast_str_and_kwarg_wrong_field(self):
        with self.assertRaises(CastError):
            _ = Simple('{"b": "bar"}', a='foo')

    def test_uncastable_str(self):
        with self.assertRaises(ParseError):
            _ = Simple('hello world')

    def test_non_string_keys(self):
        with self.assertRaises(ParseError):
            _ = Simple({1: 2})


class TestExtraFields(unittest.TestCase):

    def test_true_true_dict(self):

        class TrueTrue(
                JsonObject,
                ignore_extra=True,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(TrueTrue, 'ignore_extra'))
        self.assertTrue(TrueTrue.ignore_extra)
        self.assertTrue(hasattr(TrueTrue, 'raise_extra'))
        self.assertTrue(TrueTrue.raise_extra)
        true_true = TrueTrue({'a': 1})
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_dict(self):

        class TrueFalse(
                JsonObject,
                ignore_extra=True,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(TrueFalse, 'ignore_extra'))
        self.assertTrue(TrueFalse.ignore_extra)
        self.assertTrue(hasattr(TrueFalse, 'raise_extra'))
        self.assertFalse(TrueFalse.raise_extra)
        true_false = TrueFalse({'a': 1})
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_dict(self):

        class FalseTrue(
                JsonObject,
                ignore_extra=False,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(FalseTrue, 'ignore_extra'))
        self.assertFalse(FalseTrue.ignore_extra)
        self.assertTrue(hasattr(FalseTrue, 'raise_extra'))
        self.assertTrue(FalseTrue.raise_extra)
        with self.assertRaises(ParseError):
            _ = FalseTrue({'a': 1})

    def test_false_false_dict(self):

        class FalseFalse(
                JsonObject,
                ignore_extra=False,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(FalseFalse, 'ignore_extra'))
        self.assertFalse(FalseFalse.ignore_extra)
        self.assertTrue(hasattr(FalseFalse, 'raise_extra'))
        self.assertFalse(FalseFalse.raise_extra)
        false_false = FalseFalse({'a': 1})
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_kwarg(self):

        class TrueTrue(
                JsonObject,
                ignore_extra=True,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(TrueTrue, 'ignore_extra'))
        self.assertTrue(TrueTrue.ignore_extra)
        self.assertTrue(hasattr(TrueTrue, 'raise_extra'))
        self.assertTrue(TrueTrue.raise_extra)
        true_true = TrueTrue(a=1)
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_kwarg(self):

        class TrueFalse(
                JsonObject,
                ignore_extra=True,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(TrueFalse, 'ignore_extra'))
        self.assertTrue(TrueFalse.ignore_extra)
        self.assertTrue(hasattr(TrueFalse, 'raise_extra'))
        self.assertFalse(TrueFalse.raise_extra)
        true_false = TrueFalse(a=1)
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_kwarg(self):

        class FalseTrue(
                JsonObject,
                ignore_extra=False,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(FalseTrue, 'ignore_extra'))
        self.assertFalse(FalseTrue.ignore_extra)
        self.assertTrue(hasattr(FalseTrue, 'raise_extra'))
        self.assertTrue(FalseTrue.raise_extra)
        with self.assertRaises(ParseError):
            _ = FalseTrue(a=1)

    def test_false_false_kwarg(self):

        class FalseFalse(
                JsonObject,
                ignore_extra=False,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(FalseFalse, 'ignore_extra'))
        self.assertFalse(FalseFalse.ignore_extra)
        self.assertTrue(hasattr(FalseFalse, 'raise_extra'))
        self.assertFalse(FalseFalse.raise_extra)
        false_false = FalseFalse(a=1)
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_str(self):

        class TrueTrue(
                JsonObject,
                ignore_extra=True,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(TrueTrue, 'ignore_extra'))
        self.assertTrue(TrueTrue.ignore_extra)
        self.assertTrue(hasattr(TrueTrue, 'raise_extra'))
        self.assertTrue(TrueTrue.raise_extra)
        true_true = TrueTrue('{"a": 1}')
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_str(self):

        class TrueFalse(
                JsonObject,
                ignore_extra=True,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(TrueFalse, 'ignore_extra'))
        self.assertTrue(TrueFalse.ignore_extra)
        self.assertTrue(hasattr(TrueFalse, 'raise_extra'))
        self.assertFalse(TrueFalse.raise_extra)
        true_false = TrueFalse('{"a": 1}')
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_str(self):

        class FalseTrue(
                JsonObject,
                ignore_extra=False,
                raise_extra=True
        ):
            pass

        self.assertTrue(hasattr(FalseTrue, 'ignore_extra'))
        self.assertFalse(FalseTrue.ignore_extra)
        self.assertTrue(hasattr(FalseTrue, 'raise_extra'))
        self.assertTrue(FalseTrue.raise_extra)
        with self.assertRaises(ParseError):
            _ = FalseTrue('{"a": 1}')

    def test_false_false_str(self):

        class FalseFalse(
                JsonObject,
                ignore_extra=False,
                raise_extra=False
        ):
            pass

        self.assertTrue(hasattr(FalseFalse, 'ignore_extra'))
        self.assertFalse(FalseFalse.ignore_extra)
        self.assertTrue(hasattr(FalseFalse, 'raise_extra'))
        self.assertFalse(FalseFalse.raise_extra)
        false_false = FalseFalse('{"a": 1}')
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_extra_field_none_dict(self):

        class Respect(
                JsonObject,
                ignore_extra=False,
                raise_extra=False,
                respect_none=True
        ):
            pass

        self.assertTrue(hasattr(Respect, 'ignore_extra'))
        self.assertFalse(Respect.ignore_extra)
        self.assertTrue(hasattr(Respect, 'raise_extra'))
        self.assertFalse(Respect.raise_extra)
        self.assertTrue(hasattr(Respect, 'respect_none'))
        self.assertTrue(Respect.respect_none)
        respect = Respect({'a': None})
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_none_kwarg(self):

        class Respect(
                JsonObject,
                ignore_extra=False,
                raise_extra=False,
                respect_none=True
        ):
            pass

        self.assertTrue(hasattr(Respect, 'ignore_extra'))
        self.assertFalse(Respect.ignore_extra)
        self.assertTrue(hasattr(Respect, 'raise_extra'))
        self.assertFalse(Respect.raise_extra)
        self.assertTrue(hasattr(Respect, 'respect_none'))
        self.assertTrue(Respect.respect_none)
        respect = Respect(a=None)
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_none_str(self):

        class Respect(
                JsonObject,
                ignore_extra=False,
                raise_extra=False,
                respect_none=True
        ):
            pass

        self.assertTrue(hasattr(Respect, 'ignore_extra'))
        self.assertFalse(Respect.ignore_extra)
        self.assertTrue(hasattr(Respect, 'raise_extra'))
        self.assertFalse(Respect.raise_extra)
        self.assertTrue(hasattr(Respect, 'respect_none'))
        self.assertTrue(Respect.respect_none)
        respect = Respect('{"a": null}')
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)


if __name__ == '__main__':
    unittest.main()

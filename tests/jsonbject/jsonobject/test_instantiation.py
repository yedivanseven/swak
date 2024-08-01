import unittest
from unittest.mock import Mock
from pandas import Series
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ParseError, CastError


class Empty(JsonObject):
    pass


class Simple(JsonObject):
    a: int
    b: str


class Option(JsonObject, respect_none=True):
    a: Maybe[int](int)


class Respect(JsonObject, respect_none=True):
    a: int


class TestEmpty(unittest.TestCase):

    def test_no_arg(self):
        _ = Empty()

    def test_empty_dict(self):
        _ = Empty({})

    def test_none(self):
        _ = Empty(None)

    def test_empty_series(self):
        _ = Empty(Series())

    def test_str_of_empty_dict(self):
        _ = Empty('{}')

    def test_bytes_of_empty_dict(self):
        _ = Empty(bytes('{}'.encode('utf-8')))

    def test_bytearray_of_empty_dict(self):
        _ = Empty(bytearray('{}'.encode('utf-8')))


class TestNoneIgnored(unittest.TestCase):

    def test_dict_none(self):
        _ = Empty({'foo': None})

    def test_series_none(self):
        _ = Empty(Series({'foo': None}))

    def test_str_of_dict_none(self):
        _ = Empty("{'foo': None}")

    def test_json_of_dict_null(self):
        _ = Empty('{"foo": null}')

    def test_bytes_of_dict_null(self):
        _ = Empty(bytes('{"foo": null}'.encode('utf-8')))

    def test_bytearray_of_dict_null(self):
        _ = Empty(bytearray('{"foo": null}'.encode('utf-8')))

    def test_kwarg_none(self):
        _ = Empty(foo=None)


class TestSimple(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsInstance(obj.a, int)
        self.assertEqual(1, obj.a)
        self.assertTrue(hasattr(obj, 'b'))
        self.assertIsInstance(obj.b, str)
        self.assertEqual('foo', obj.b)

    def test_dict(self):
        simple = Simple({'a': 1, 'b': 'foo'})
        self.check_attributes(simple)

    def test_dict_and_kwarg(self):
        simple = Simple({'b': 'foo'}, a=1)
        self.check_attributes(simple)

    def test_kwargs(self):
        simple = Simple(a=1, b='foo')
        self.check_attributes(simple)

    def test_empty_dict_and_kwargs(self):
        simple = Simple({}, a=1, b='foo')
        self.check_attributes(simple)

    def test_empty_series_and_kwargs(self):
        simple = Simple(Series(), a=1, b='foo')
        self.check_attributes(simple)

    def test_none_and_kwargs(self):
        simple = Simple(None, a=1, b='foo')
        self.check_attributes(simple)

    def test_json(self):
        simple = Simple('{"a": 1, "b": "foo"}')
        self.check_attributes(simple)

    def test_json_and_kwarg(self):
        simple = Simple('{"b": "foo"}', a=1)
        self.check_attributes(simple)

    def test_str(self):
        simple = Simple("{'a': 1, 'b': 'foo'}")
        self.check_attributes(simple)

    def test_str_and_kwarg(self):
        simple = Simple("{'b': 'foo'}", a=1)
        self.check_attributes(simple)

    def test_bytes(self):
        simple = Simple(bytes('{"a": 1, "b": "foo"}'.encode('utf-8')))
        self.check_attributes(simple)

    def test_bytes_and_kwarg(self):
        simple = Simple(bytes('{"b": "foo"}'.encode('utf-8')), a=1)
        self.check_attributes(simple)

    def test_bytearray(self):
        simple = Simple(bytearray('{"a": 1, "b": "foo"}'.encode('utf-8')))
        self.check_attributes(simple)

    def test_bytearray_and_kwarg(self):
        simple = Simple(bytearray('{"b": "foo"}'.encode('utf-8')), a=1)
        self.check_attributes(simple)

    def test_series(self):
        simple = Simple(Series({'a': 1, 'b': 'foo'}))
        self.check_attributes(simple)

    def test_series_and_kwarg(self):
        simple = Simple(Series({'b': 'foo'}), a=1)
        self.check_attributes(simple)

    def test_self(self):
        simple = Simple({'a': 1, 'b': 'foo'})
        derived = Simple(simple)
        self.check_attributes(derived)

    def test_kwarg_trumps_dict(self):
        simple = Simple({'a': 1, 'b': 'bar'}, b='foo')
        self.check_attributes(simple)

    def test_kwarg_trumps_json(self):
        simple = Simple('{"a": 1, "b": "bar"}', b='foo')
        self.check_attributes(simple)

    def test_kwarg_trumps_str(self):
        simple = Simple("{'a': 1, 'b': 'bar'}", b='foo')
        self.check_attributes(simple)

    def test_kwarg_trumps_bytes(self):
        simple = Simple(bytes('{"a": 1, "b": "bar"}'.encode('utf8')), b='foo')
        self.check_attributes(simple)

    def test_kwarg_trumps_bytearray(self):
        simple = Simple(
            bytearray('{"a": 1, "b": "bar"}'.encode('utf8')),
            b='foo'
        )
        self.check_attributes(simple)

    def test_kwarg_trumps_series(self):
        simple = Simple(Series({'a': 1, 'b': 'bar'}), b='foo')
        self.check_attributes(simple)

    def test_kwarg_trumps_self(self):
        simple = Simple({'a': 1, 'b': 'bar'})
        derived = Simple(simple, b='foo')
        self.check_attributes(derived)

    def test_kwarg_none_ignored(self):
        simple = Simple({'a': 1, 'b': 'foo'}, b=None)
        self.check_attributes(simple)

    def test_extra_kwarg_none_ignored(self):
        simple = Simple({'a': 1, 'b': 'foo'}, c=None)
        self.check_attributes(simple)


class TestNone(unittest.TestCase):

    def check_attribute(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsNone(obj.a)

    def test_dict(self):
        option = Option({'a': None})
        self.check_attribute(option)

    def test_null_str_dict(self):
        option = Option({'a': 'null'})
        self.check_attribute(option)

    def test_none_str_dict(self):
        option = Option({'a': 'None'})
        self.check_attribute(option)

    def test_json(self):
        option = Option('{"a": null}')
        self.check_attribute(option)

    def test_json_none(self):
        option = Option('{"a": None}')
        self.check_attribute(option)

    def test_json_null_str(self):
        option = Option('{"a": "null"}')
        self.check_attribute(option)

    def test_json_none_str(self):
        option = Option('{"a": "None"}')
        self.check_attribute(option)

    def test_str(self):
        option = Option("{'a': None}")
        self.check_attribute(option)

    def test_str_null_str(self):
        option = Option("{'a': 'null'}")
        self.check_attribute(option)

    def test_str_none_str(self):
        option = Option("{'a': 'None'}")
        self.check_attribute(option)

    def test_series(self):
        option = Option(Series({'a': None}))
        self.check_attribute(option)

    def test_series_null_str(self):
        option = Option(Series({'a': 'null'}))
        self.check_attribute(option)

    def test_series_none_str(self):
        option = Option(Series({'a': 'None'}))
        self.check_attribute(option)

    def test_self(self):
        option = Option(Option({'a': None}))
        self.check_attribute(option)

    def test_kwarg(self):
        option = Option(a=None)
        self.check_attribute(option)

    def test_null_str_kwarg(self):
        option = Option(a='null')
        self.check_attribute(option)

    def test_none_str_kwarg(self):
        option = Option(a='None')
        self.check_attribute(option)

    def test_empty_dict_and_kwarg(self):
        option = Option({}, a=None)
        self.check_attribute(option)

    def test_none_and_kwarg(self):
        option = Option(None, a=None)
        self.check_attribute(option)

    def test_kwarg_trumps_dict(self):
        option = Option({'a': 2}, a=None)
        self.check_attribute(option)


class TestTypeCasting(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertIsInstance(obj.a, int)
        self.assertEqual(1.0, obj.a)
        self.assertIsInstance(obj.b, str)
        self.assertEqual('2.0', obj.b)

    def test_type_cast_called(self):
        mock = Mock()

        class Custom(JsonObject):
            a: mock

        _ = Custom(a=1)
        mock.assert_called_once()
        mock.assert_called_once_with(1)

    def test_dict(self):
        simple = Simple({'a': 1.0, 'b': 2.0})
        self.check_attributes(simple)

    def test_dict_and_kwarg(self):
        simple = Simple({'b': 2.0}, a=1.0)
        self.check_attributes(simple)

    def test_kwargs(self):
        simple = Simple(a=1.0, b=2.0)
        self.check_attributes(simple)

    def test_empty_and_kwargs(self):
        simple = Simple({}, a=1.0, b=2.0)
        self.check_attributes(simple)

    def test_none_and_kwargs(self):
        simple = Simple(None, a=1.0, b=2.0)
        self.check_attributes(simple)

    def test_json(self):
        simple = Simple('{"a": 1.0, "b": 2.0}')
        self.check_attributes(simple)

    def test_json_and_kwarg(self):
        simple = Simple('{"b": 2.0}', a=1.0)
        self.check_attributes(simple)

    def test_str(self):
        simple = Simple("{'a': 1.0, 'b': 2.0}")
        self.check_attributes(simple)

    def test_str_and_kwarg(self):
        simple = Simple("{'b': 2.0}", a=1.0)
        self.check_attributes(simple)

    def test_bytes(self):
        simple = Simple(bytes('{"a": 1.0, "b": 2.0}'.encode('utf-8')))
        self.check_attributes(simple)

    def test_bytes_and_kwarg(self):
        simple = Simple(bytes('{"b": 2.0}'.encode('utf-8')), a=1.0)
        self.check_attributes(simple)

    def test_bytearray(self):
        simple = Simple(bytearray('{"a": 1.0, "b": 2.0}'.encode('utf-8')))
        self.check_attributes(simple)

    def test_bytearray_and_kwarg(self):
        simple = Simple(bytearray('{"b": 2.0}'.encode('utf-8')), a=1.0)
        self.check_attributes(simple)

    def test_series(self):
        simple = Simple(Series({'a': 1.0, 'b': 2.0}))
        self.check_attributes(simple)

    def test_series_and_kwarg(self):
        simple = Simple(Series({'b': 2.0}), a=1.0)
        self.check_attributes(simple)


class TestExceptions(unittest.TestCase):

    def test_parse_dict_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple({'b': 'foo'})
        self.assertEqual(expected, str(error.exception))

    def test_parse_kwargs_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple(b='foo')
        self.assertEqual(expected, str(error.exception))

    def test_parse_json_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple('{"b": "foo"}')
        self.assertEqual(expected, str(error.exception))

    def test_parse_str_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple("{'b': 'foo'}")
        self.assertEqual(expected, str(error.exception))

    def test_parse_bytes_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple(bytes('{"b": "foo"}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_parse_bytearray_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple(bytearray('{"b": "foo"}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_parse_series_missing_fields(self):
        expected = "Missing non-default fields ['a']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple(Series({'b': 'foo'}))
        self.assertEqual(expected, str(error.exception))

    def test_uncastable_str(self):
        expected = "Could not parse hello world as JSON!"
        with self.assertRaises(ParseError) as error:
            _ = Simple('hello world')
        self.assertEqual(expected, str(error.exception))

    def test_non_string_keys(self):
        expected = "Missing non-default fields ['a', 'b']!"
        with self.assertRaises(ParseError) as error:
            _ = Simple({1: 2})
        self.assertEqual(expected, str(error.exception))

    def test_cast_dict_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple({'a': 'foo', 'b': 'bar'})
        self.assertEqual(expected, str(error.exception))

    def test_cast_dict_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple({'b': 'bar'}, a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_kwargs_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(b='bar', a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_str_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple("{'a': 'foo', 'b': 'bar'}")
        self.assertEqual(expected, str(error.exception))

    def test_cast_str_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple("{'b': 'bar'}", a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_json_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple('{"a": "foo", "b": "bar"}')
        self.assertEqual(expected, str(error.exception))

    def test_cast_json_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple('{"b": "bar"}', a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_bytes_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(bytes('{"a": "foo", "b": "bar"}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_cast_bytes_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(bytes('{"b": "bar"}'.encode('utf-8')), a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_bytearray_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(bytearray('{"a": "foo", "b": "bar"}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_cast_bytearray_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(bytearray('{"b": "bar"}'.encode('utf-8')), a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_series_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(Series({'a': 'foo', 'b': 'bar'}))
        self.assertEqual(expected, str(error.exception))

    def test_cast_series_and_kwarg_wrong_field(self):
        expected = "Could not cast JSON fields ['a']!"
        with self.assertRaises(CastError) as error:
            _ = Simple(Series({'b': 'bar'}), a='foo')
        self.assertEqual(expected, str(error.exception))

    def test_cast_non_maybe_none_fields(self):
        expected = ("For fields ['a'] to be None, mark them"
                    " as Maybe(<YOUR_TYPE>) in the schema!")
        with self.assertRaises(CastError) as error:
            _ = Respect(a=None)
        self.assertEqual(expected, str(error.exception))


class TestExtraFields(unittest.TestCase):

    class TrueTrue(JsonObject, ignore_extra=True, raise_extra=True):
        pass

    class TrueFalse(JsonObject, ignore_extra=True, raise_extra=False):
        pass

    class FalseTrue(JsonObject, ignore_extra=False, raise_extra=True):
        pass

    class FalseFalse(JsonObject, ignore_extra=False, raise_extra=False):
        pass

    class Respect(
        JsonObject,
        ignore_extra=False,
        raise_extra=False,
        respect_none=True
    ):
        pass

    def test_true_true_dict(self):
        true_true = self.TrueTrue({'a': 1})
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_dict(self):
        true_false = self.TrueFalse({'a': 1})
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_dict(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self. FalseTrue({'a': 1})
        self.assertEqual(expected, str(error.exception))

    def test_false_false_dict(self):
        false_false = self.FalseFalse({'a': 1})
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_kwarg(self):
        true_true = self.TrueTrue(a=1)
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_kwarg(self):
        true_false = self.TrueFalse(a=1)
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_kwarg(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self.FalseTrue(a=1)
        self.assertEqual(expected, str(error.exception))

    def test_false_false_kwarg(self):
        false_false = self.FalseFalse(a=1)
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_str(self):
        true_true = self.TrueTrue("{'a': 1}")
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_str(self):
        true_false = self.TrueFalse("{'a': 1}")
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_str(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self.FalseTrue("{'a': 1}")
        self.assertEqual(expected, str(error.exception))

    def test_false_false_str(self):
        false_false = self.FalseFalse("{'a': 1}")
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_json(self):
        true_true = self.TrueTrue('{"a": 1}')
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_json(self):
        true_false = self.TrueFalse('{"a": 1}')
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_json(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self.FalseTrue('{"a": 1}')
        self.assertEqual(expected, str(error.exception))

    def test_false_false_json(self):
        false_false = self.FalseFalse('{"a": 1}')
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_bytes(self):
        true_true = self.TrueTrue(bytes('{"a": 1}'.encode('utf-8')))
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_bytes(self):
        true_false = self.TrueFalse(bytes('{"a": 1}'.encode('utf-8')))
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_bytes(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self.FalseTrue(bytes('{"a": 1}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_false_false_bytes(self):
        false_false = self.FalseFalse(bytes('{"a": 1}'.encode('utf-8')))
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_bytearray(self):
        true_true = self.TrueTrue(bytearray('{"a": 1}'.encode('utf-8')))
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_bytearray(self):
        true_false = self.TrueFalse(bytearray('{"a": 1}'.encode('utf-8')))
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_bytearray(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self.FalseTrue(bytearray('{"a": 1}'.encode('utf-8')))
        self.assertEqual(expected, str(error.exception))

    def test_false_false_bytearray(self):
        false_false = self.FalseFalse(bytearray('{"a": 1}'.encode('utf-8')))
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_true_true_series(self):
        true_true = self.TrueTrue(Series({'a': 1}))
        self.assertFalse(hasattr(true_true, 'a'))

    def test_true_false_series(self):
        true_false = self.TrueFalse(Series({'a': 1}))
        self.assertFalse(hasattr(true_false, 'a'))

    def test_false_true_series(self):
        expected = "Fields {'a'} are not in schema!"
        with self.assertRaises(ParseError) as error:
            _ = self. FalseTrue(Series({'a': 1}))
        self.assertEqual(expected, str(error.exception))

    def test_false_false_series(self):
        false_false = self.FalseFalse((Series({'a': 1})))
        self.assertTrue(hasattr(false_false, 'a'))
        self.assertEqual(1, false_false.a)

    def test_extra_field_none_dict(self):
        respect = self.Respect({'a': None})
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_none_kwarg(self):
        respect = self.Respect(a=None)
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_none_str(self):
        respect = self.Respect("{'a': None}")
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_null_json(self):
        respect = self.Respect('{"a": null}')
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_null_bytes(self):
        respect = self.Respect(bytes('{"a": null}'.encode('utf-8')))
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_null_bytearray(self):
        respect = self.Respect(bytearray('{"a": null}'.encode('utf-8')))
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_field_none_series(self):
        respect = self.Respect(Series({'a': None}))
        self.assertTrue(hasattr(respect, 'a'))
        self.assertIsNone(respect.a)

    def test_extra_raises_on_non_string_keys(self):
        expected = 'Extra fields must have string keys!'
        with self.assertRaises(ParseError) as error:
            _ = self.FalseFalse({0: 1})
        self.assertEqual(expected, str(error.exception))

    def test_extra_raises_on_non_identifier_keys(self):
        expected = 'Keys must be (dot.separated) valid python identifiers!'
        with self.assertRaises(ParseError) as error:
            _ = self.FalseFalse({' 034- 5 / ': 1})
        self.assertEqual(expected, str(error.exception))


if __name__ == '__main__':
    unittest.main()

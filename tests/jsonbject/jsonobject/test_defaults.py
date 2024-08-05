import unittest
from unittest.mock import Mock
from pandas import Series
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import ValidationErrors


class Default(JsonObject):
    a: int = 1


class NoneDefault(JsonObject):
    a: Maybe[int](int) = None


class Respect(JsonObject, respect_none=True):
    a: Maybe[int](int) = 1


class TestDefault(unittest.TestCase):

    def check_attribute(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsInstance(obj.a, int)
        self.assertEqual(1, obj.a)

    def test_empty(self):
        default = Default()
        self.check_attribute(default)

    def test_empty_dict(self):
        default = Default({})
        self.check_attribute(default)

    def test_None(self):
        default = Default(None)
        self.check_attribute(default)

    def test_dict_none(self):
        default = Default({'a': None})
        self.check_attribute(default)

    def test_empty_series(self):
        default = Default(Series())
        self.check_attribute(default)

    def test_series_none(self):
        default = Default(Series({'a': None}))
        self.check_attribute(default)

    def test_str_of_empty_dict(self):
        default = Default('{}')
        self.check_attribute(default)

    def test_str_of_dict_none(self):
        default = Default("{'a': None}")
        self.check_attribute(default)

    def test_json_of_dict_null(self):
        default = Default('{"a": null}')
        self.check_attribute(default)

    def test_bytes_of_empty_dict(self):
        default = Default(bytes('{}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytes_of_dict_null(self):
        default = Default(bytes('{"a": null}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytearray_of_empty_dict(self):
        default = Default(bytearray('{}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytearray_of_dict_null(self):
        default = Default(bytearray('{"a": null}'.encode('utf-8')))
        self.check_attribute(default)

    def test_kwarg_none(self):
        default = Default(a=None)
        self.check_attribute(default)


class TestDefaultOverwritten(unittest.TestCase):

    def check_attribute(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsInstance(obj.a, int)
        self.assertEqual(42, obj.a)

    def test_dict(self):
        default = Default({'a': 42})
        self.check_attribute(default)

    def test_kwarg(self):
        default = Default(a=42)
        self.check_attribute(default)

    def test_series(self):
        default = Default(Series({'a': 42}))
        self.check_attribute(default)

    def test_str(self):
        default = Default("{'a': 42}")
        self.check_attribute(default)

    def test_json(self):
        default = Default('{"a": 42}')
        self.check_attribute(default)

    def test_bytes(self):
        default = Default(bytes('{"a": 42}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytearray(self):
        default = Default(bytearray('{"a": 42}'.encode('utf-8')))
        self.check_attribute(default)


class TestNoneDefaultOverwritten(unittest.TestCase):

    def check_attribute(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsInstance(obj.a, int)
        self.assertEqual(42, obj.a)

    def test_dict(self):
        default = NoneDefault({'a': 42})
        self.check_attribute(default)

    def test_kwarg(self):
        default = NoneDefault(a=42)
        self.check_attribute(default)

    def test_series(self):
        default = NoneDefault(Series({'a': 42}))
        self.check_attribute(default)

    def test_str(self):
        default = NoneDefault("{'a': 42}")
        self.check_attribute(default)

    def test_json(self):
        default = NoneDefault('{"a": 42}')
        self.check_attribute(default)

    def test_bytes(self):
        default = NoneDefault(bytes('{"a": 42}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytearray(self):
        default = NoneDefault(bytearray('{"a": 42}'.encode('utf-8')))
        self.check_attribute(default)


class TestDefaultOverwrittenWithNone(unittest.TestCase):

    def check_attribute(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertIsNone(obj.a)

    def test_dict(self):
        default = Respect({'a': None})
        self.check_attribute(default)

    def test_kwarg(self):
        default = Respect(a=None)
        self.check_attribute(default)

    def test_kwarg_null_str(self):
        default = Respect(a='null')
        self.check_attribute(default)

    def test_kwarg_none_str(self):
        default = Respect(a='None')
        self.check_attribute(default)

    def test_series(self):
        default = Respect(Series({'a': None}))
        self.check_attribute(default)

    def test_str(self):
        default = Respect("{'a': None}")
        self.check_attribute(default)

    def test_json(self):
        default = Respect('{"a": null}')
        self.check_attribute(default)

    def test_bytes(self):
        default = Respect(bytes('{"a": null}'.encode('utf-8')))
        self.check_attribute(default)

    def test_bytearray(self):
        default = Respect(bytearray('{"a": null}'.encode('utf-8')))
        self.check_attribute(default)


class TestException(unittest.TestCase):

    def test_dict(self):
        with self.assertRaises(ValidationErrors):
            _ = Default({'a': 'bar'})

    def test_kwarg(self):
        with self.assertRaises(ValidationErrors):
            _ = Default(a='bar')

    def test_series(self):
        with self.assertRaises(ValidationErrors):
            _ = Default(Series({'a': 'bar'}))

    def test_str(self):
        with self.assertRaises(ValidationErrors):
            _ = Default("{'a': 'bar'}")

    def test_json(self):
        with self.assertRaises(ValidationErrors):
            _ = Default('{"a": "bar"}')

    def test_bytes(self):
        with self.assertRaises(ValidationErrors):
            _ = Default(bytes('{"a": "bar"}'.encode('utf-8')))

    def test_bytearray(self):
        with self.assertRaises(ValidationErrors):
            _ = Default(bytearray('{"a": "bar"}'.encode('utf-8')))


class TestTypeCast(unittest.TestCase):

    def test_cast_works(self):

        class Custom(JsonObject):
            a: str = 1.0

        custom = Custom()
        self.assertTrue(hasattr(custom, 'a'))
        self.assertEqual('1.0', custom.a)

    def test_caster_called(self):
        mock = Mock()
        mock.return_value = 2.0

        class Custom(JsonObject):
            a: mock = 1.0

        _ = Custom()
        mock.assert_called()
        self.assertEqual(2, mock.call_count)
        ((first,), _), ((second,), _) = mock.call_args_list
        self.assertEqual(1.0, first)
        self.assertEqual(2.0, second)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import Mock
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe
from swak.jsonobject.exceptions import CastError


class Default(JsonObject):
    a: int = 1
    b: str


class TestInstantiation(unittest.TestCase):

    def test_dict(self):
        default = Default({'b': 'foo'})
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(1, default.a)

    def test_kwarg(self):
        default = Default(b='foo')
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(1, default.a)

    def test_str(self):
        default = Default('{"b": "foo"}')
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(1, default.a)

    def test_dict_trumps_default(self):
        default = Default({'a': 2, 'b': 'foo'})
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(2, default.a)

    def test_kwarg_trumps_default(self):
        default = Default(a=2, b='foo')
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(2, default.a)

    def test_str_trumps_default(self):
        default = Default('{"a": 2, "b": "foo"}')
        self.assertTrue(hasattr(default, 'a'))
        self.assertIsInstance(default.a, int)
        self.assertEqual(2, default.a)


class TestException(unittest.TestCase):

    def test_dict(self):
        with self.assertRaises(CastError):
            _ = Default({'b': 'foo', 'a': 'bar'})

    def test_kwarg(self):
        with self.assertRaises(CastError):
            _ = Default({'b': 'foo'}, a='bar')

    def test_str(self):
        with self.assertRaises(CastError):
            _ = Default('{"b": "foo", "a": "bar"}')


class TestTypeCast(unittest.TestCase):

    def test_cast_works(self):

        class Custom(JsonObject):
            a: str = 1.0

        custom = Custom()
        self.assertTrue(hasattr(custom, 'a'))
        self.assertIsInstance(custom.a, str)
        self.assertEqual('1.0', custom.a)

    def test_caster_called(self):
        mock = Mock()
        mock.return_value = 2.0

        class Custom(JsonObject):
            a: mock = 1.0

        _ = Custom()
        mock.assert_called()
        self.assertEqual(2, mock.call_count)
        (first, _), (second, _) = mock.call_args_list
        self.assertEqual(1.0, first[0])
        self.assertEqual(2.0, second[0])


class TestMaybe(unittest.TestCase):

    class NoneDefault(JsonObject):
        a: Maybe[int](int) = None
        b: str = 'foo'

    class MaybeDefault(JsonObject, respect_none=True):
        a: Maybe[int](int) = 1
        b: str = 'foo'

    def test_none_default_remains_none(self):
        none_default = self.NoneDefault()
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsNone(none_default.a)

    def test_none_default_is_overridden_by_dict(self):
        none_default = self.NoneDefault({'a': 1})
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(1, none_default.a)

    def test_none_default_is_overridden_by_kwarg(self):
        none_default = self.NoneDefault(a=1)
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(1, none_default.a)

    def test_none_default_is_overridden_by_dict_kwarg(self):
        none_default = self.NoneDefault({'a': 1}, a=2)
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(2, none_default.a)

    def test_none_default_is_overridden_by_str(self):
        none_default = self.NoneDefault('{"a": 1}')
        self.assertTrue(hasattr(none_default, 'a'))
        self.assertIsInstance(none_default.a, int)
        self.assertEqual(1, none_default.a)

    def test_none_dict_overwrites_default(self):
        maybe_default = self.MaybeDefault({'a': None})
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_kwarg_overwrites_default(self):
        maybe_default = self.MaybeDefault(a=None)
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_dict_kwarg_overwrites_default(self):
        maybe_default = self.MaybeDefault({'a': 2}, a=None)
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)

    def test_none_str_overwrites_default(self):
        maybe_default = self.MaybeDefault('{"a": null}')
        self.assertTrue(hasattr(maybe_default, 'a'))
        self.assertIsNone(maybe_default.a)


if __name__ == '__main__':
    unittest.main()

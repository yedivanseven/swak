import unittest
from unittest.mock import Mock
import pandas as pd
from swak.jsonobject import JsonObject
from swak.jsonobject.exceptions import ParseError, CastError


class Empty(JsonObject):
    pass


class ParentOfEmpty(JsonObject):
    e: Empty


class Child(JsonObject):
    b: str


class MockChild(JsonObject):
    b: str

    @property
    def as_json(self):
        return 'mocked as_json'

    @property
    def as_dtype(self):
        return 'mocked as_dtype'


class IntChild(JsonObject):
    b: int


class Parent(JsonObject):
    a: int
    child: Child


class MockParent(JsonObject):
    a: int
    child: MockChild


class IntParent(Parent):
    a: int
    child: IntChild


class IgnoreChild(Child, ignore_extra=True, raise_extra=False):
    pass


class IgnoreParent(Parent):
    child: IgnoreChild


class AllowChild(Child, ignore_extra=False, raise_extra=False):
    pass


class AllowParent(Parent):
    child: AllowChild


class TestInstantiation(unittest.TestCase):

    def test_dict(self):
        parent = Parent(a=1, child={'b': 'foo'})
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('foo', parent.child.b)

    def test_str(self):
        parent = Parent(a=1, child='{"b":"foo"}')
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('foo', parent.child.b)

    def test_self(self):
        child = Child({'b': 'foo'})
        parent = Parent(a=1, child=child)
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('foo', parent.child.b)

    def test_dict_kwarg_trumps_dict(self):
        parent = Parent({'a': 1, 'child': {'b': 'foo'}}, child={'b': 'bar'})
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('bar', parent.child.b)

    def test_none_kwarg_ignored(self):
        parent = Parent({'a': 1, 'child': {'b': 'foo'}}, child=None)
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('foo', parent.child.b)

    def test_dict_kwarg_trumps_str(self):
        parent = Parent('{"a": 1, "child": {"b": "foo"}}', child={'b': 'bar'})
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('bar', parent.child.b)

    def test_str_kwarg_trumps_dict(self):
        parent = Parent({'a': 1, 'child': {'b': 'foo'}}, child='{"b": "bar"}')
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('bar', parent.child.b)

    def test_str_kwarg_trumps_str(self):
        parent = Parent(
            '{"a": 1, "child": {"b": "foo"}}',
            child='{"b": "bar"}'
        )
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.b, str)
        self.assertEqual('bar', parent.child.b)

    def test_empty_child(self):
        parent = ParentOfEmpty(e={})
        self.assertTrue(hasattr(parent, 'e'))
        self.assertIsInstance(parent.e, Empty)


class TestViews(unittest.TestCase):

    def test_str(self):
        parent = Parent(a=1, child={'b': 'foo'})
        child = Child({'b': 'foo'})
        expected = f'{{"a": 1, "child": {str(child)}}}'
        self.assertEqual(str(parent), expected)

    def test_as_json_called_by_str(self):
        parent = MockParent(a=1, child={'b': 'foo'})
        expected = '{"a": 1, "child": "mocked as_json"}'
        self.assertEqual(str(parent), expected)

    def test_repr(self):
        parent = Parent(a=1, child={'b': 'foo'})
        exp = '{\n    "a": 1,\n    "child": {\n        "b": "foo"\n    }\n}'
        self.assertEqual(repr(parent), exp)

    def test_as_json_called_by_repr(self):
        parent = MockParent(a=1, child={'b': 'foo'})
        exp = '{\n    "a": 1,\n    "child": "mocked as_json"\n}'
        self.assertEqual(repr(parent), exp)

    def test_as_json(self):
        parent = Parent(a=1, child={'b': 'foo'})
        expected = {'a': 1, 'child': {'b': 'foo'}}
        self.assertIsInstance(parent.as_json, dict)
        self.assertDictEqual(expected, parent.as_json)

    def test_as_json_called_by_as_json(self):
        parent = MockParent(a=1, child={'b': 'foo'})
        expected = {'a': 1, 'child': 'mocked as_json'}
        self.assertIsInstance(parent.as_json, dict)
        self.assertDictEqual(expected, parent.as_json)

    def test_as_dtype(self):
        parent = Parent(a=1, child={'b': 'foo'})
        self.assertIsInstance(parent.as_dtype, str)
        expected = '{"a": 1, "child": {"b": "foo"}}'
        self.assertEqual(expected, parent.as_dtype)

    def test_as_json_called_by_as_dtype(self):
        parent = MockParent(a=1, child={'b': 'foo'})
        self.assertIsInstance(parent.as_dtype, str)
        expected = '{"a": 1, "child": "mocked as_json"}'
        self.assertEqual(expected, parent.as_dtype)

    def test_as_series(self):
        parent = Parent(a=1, child={'b': 'foo'})
        self.assertIsInstance(parent.as_series, pd.Series)
        expected = pd.Series({'a': 1, 'child': '{"b": "foo"}'}, name='Parent')
        pd.testing.assert_series_equal(expected, parent.as_series)
        self.assertIsInstance(parent.as_series.child, str)
        self.assertEqual('{"b": "foo"}', parent.as_series.child)

    def test_as_dtype_called_by_as_series(self):
        parent = MockParent(a=1, child={'b': 'foo'})
        self.assertIsInstance(parent.as_series, pd.Series)
        expected = pd.Series(
            {'a': 1, 'child': 'mocked as_dtype'},
            name='MockParent'
        )
        pd.testing.assert_series_equal(expected, parent.as_series)
        self.assertIsInstance(parent.as_series.child, str)
        self.assertEqual(parent.as_series.child, 'mocked as_dtype')


class TestExceptionRaised(unittest.TestCase):

    def test_parse_dict_missing_fields(self):
        with self.assertRaises(ParseError):
            _ = Parent(a=1, child={})

    def test_parse_str_missing_fields(self):
        with self.assertRaises(ParseError):
            _ = Parent(a=1, child='{}')

    def test_cast_dict_wrong_field(self):
        with self.assertRaises(CastError):
            _ = IntParent(a=1, child={'b': 'foo'})

    def test_cast_str_wrong_field(self):
        with self.assertRaises(CastError):
            _ = IntParent(a=1, child='{"b": "foo"}')

    def test_uncastable_str(self):
        with self.assertRaises(ParseError):
            _ = Parent(a=1, child='hello world')

    def test_non_string_keys(self):
        with self.assertRaises(ParseError):
            _ = Parent(a=1, child={1: 'foo'})


class TestDeepNesting(unittest.TestCase):

    def test_instantiation(self):
        mock = Mock()

        class Child(JsonObject):
            c: mock

        class Parent(JsonObject):
            b: Child

        class Grand(JsonObject):
            a: Parent

        _ = Grand({'a': {'b': {'c': 1}}})
        mock.assert_called_once()
        mock.assert_called_once_with(1)

    def test_value(self):

        class Child(JsonObject):
            c: int

        class Parent(JsonObject):
            b: Child

        class Grand(JsonObject):
            a: Parent

        grand = Grand({'a': {'b': {'c': 3}}})
        self.assertIsInstance(grand.a.b, Child)
        self.assertTrue(hasattr(grand.a.b, 'c'))
        self.assertIsInstance(grand.a.b.c, int)
        self.assertEqual(3, grand.a.b.c)


class TestExtraFields(unittest.TestCase):

    def setUp(self) -> None:
        self.inital = {
            'a': 1,
            'child': {
                'b': 'foo',
                'c': 2.0
            }
        }

    def test_raises(self):
        with self.assertRaises(ParseError):
            _ = Parent(self.inital)

    def test_ignore(self):
        parent = IgnoreParent(self.inital)
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertFalse(hasattr(parent.child, 'c'))

    def test_allow(self):
        parent = AllowParent(self.inital)
        self.assertTrue(hasattr(parent, 'child'))
        self.assertIsInstance(parent.child, Child)
        self.assertTrue(hasattr(parent.child, 'b'))
        self.assertIsInstance(parent.child.c, float)
        self.assertEqual(2.0, parent.child.c)


if __name__ == '__main__':
    unittest.main()

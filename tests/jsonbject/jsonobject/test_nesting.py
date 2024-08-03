import unittest
import pandas as pd
from swak.jsonobject import JsonObject
from swak.jsonobject.exceptions import ParseError


class Child(JsonObject):
    b: float = 42.0
    c: int


class Parent(JsonObject):
    a: str = 'foo'
    child: Child


class TestInstantiationDict(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(42.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(1, obj.child.c)

    def test_dict_dict(self):
        p = Parent({'child': {'c': 1}})
        self.check_attributes(p)

    def test_dict_self(self):
        p = Parent({'child': Child({'c': 1})})
        self.check_attributes(p)

    def test_dict_series(self):
        p = Parent({'child': pd.Series({'c': 1})})
        self.check_attributes(p)

    def test_dict_str(self):
        p = Parent({'child': "{'c': 1}"})
        self.check_attributes(p)

    def test_dict_json(self):
        p = Parent({'child': '{"c": 1}'})
        self.check_attributes(p)

    def test_dict_bytes(self):
        p = Parent({'child': bytes('{"c": 1}'.encode('utf-8'))})
        self.check_attributes(p)

    def test_dict_bytearray(self):
        p = Parent({'child': bytearray('{"c": 1}'.encode('utf-8'))})
        self.check_attributes(p)

    def test_nested_error(self):
        with self.assertRaises(ExceptionGroup) as eg:
            _ = Parent({'d': 1, 'child': {'b': None, 'e': 2}})
        self.assertIsInstance(eg.exception.exceptions[0], ExceptionGroup)
        self.assertIsInstance(eg.exception.exceptions[1], ParseError)


class TestInstantiationKwarg(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(42.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(1, obj.child.c)

    def test_kwarg_dict(self):
        p = Parent(child={'c': 1})
        self.check_attributes(p)

    def test_kwarg_self(self):
        p = Parent(child=Child({'c': 1}))
        self.check_attributes(p)

    def test_kwarg_series(self):
        p = Parent(child=pd.Series({'c': 1}))
        self.check_attributes(p)

    def test_kwarg_str(self):
        p = Parent(child="{'c': 1}")
        self.check_attributes(p)

    def test_kwarg_json(self):
        p = Parent(child='{"c": 1}')
        self.check_attributes(p)

    def test_kwarg_bytes(self):
        p = Parent(child=bytes('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_kwarg_bytearray(self):
        p = Parent(child=bytearray('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_empty_kwarg_dict(self):
        p = Parent({}, child={'c': 1})
        self.check_attributes(p)

    def test_empty_kwarg_self(self):
        p = Parent({}, child=Child({'c': 1}))
        self.check_attributes(p)

    def test_empty_kwarg_series(self):
        p = Parent({}, child=pd.Series({'c': 1}))
        self.check_attributes(p)

    def test_empty_kwarg_str(self):
        p = Parent({}, child="{'c': 1}")
        self.check_attributes(p)

    def test_empty_kwarg_json(self):
        p = Parent({}, child='{"c": 1}')
        self.check_attributes(p)

    def test_empty_kwarg_bytes(self):
        p = Parent({}, child=bytes('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_empty_kwarg_bytearray(self):
        p = Parent({}, child=bytearray('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_none_kwarg_dict(self):
        p = Parent(None, child={'c': 1})
        self.check_attributes(p)

    def test_none_kwarg_self(self):
        p = Parent(None, child=Child({'c': 1}))
        self.check_attributes(p)

    def test_none_kwarg_series(self):
        p = Parent(None, child=pd.Series({'c': 1}))
        self.check_attributes(p)

    def test_none_kwarg_str(self):
        p = Parent(None, child="{'c': 1}")
        self.check_attributes(p)

    def test_none_kwarg_json(self):
        p = Parent(None, child='{"c": 1}')
        self.check_attributes(p)

    def test_none_kwarg_bytes(self):
        p = Parent(None, child=bytes('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_none_kwarg_bytearray(self):
        p = Parent(None, child=bytearray('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)


class TestInstantiationSeries(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(42.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(1, obj.child.c)

    def test_series_dict(self):
        p = Parent(pd.Series({'child': {'c': 1}}))
        self.check_attributes(p)

    def test_series_self(self):
        p = Parent(pd.Series({'child': Child({'c': 1})}))
        self.check_attributes(p)

    def test_series_series(self):
        p = Parent(pd.Series({'child': pd.Series({'c': 1})}))
        self.check_attributes(p)

    def test_series_str(self):
        p = Parent(pd.Series({'child': "{'c': 1}"}))
        self.check_attributes(p)

    def test_series_json(self):
        p = Parent(pd.Series({'child': '{"c": 1}'}))
        self.check_attributes(p)

    def test_series_bytes(self):
        p = Parent(pd.Series({'child': bytes('{"c": 1}'.encode('utf-8'))}))
        self.check_attributes(p)

    def test_series_bytearray(self):
        p = Parent(pd.Series({'child': bytearray('{"c": 1}'.encode('utf-8'))}))
        self.check_attributes(p)


class TestInstantiationOther(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(42.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(1, obj.child.c)

    def test_str(self):
        p = Parent("{'child': {'c': 1}}")
        self.check_attributes(p)

    def test_json(self):
        p = Parent('{"child": {"c": 1}}')
        self.check_attributes(p)

    def test_bytes(self):
        p = Parent(bytes('{"child": {"c": 1}}'.encode('utf-8')))
        self.check_attributes(p)

    def test_bytearray(self):
        p = Parent(bytearray('{"child": {"c": 1}}'.encode('utf-8')))
        self.check_attributes(p)


class TestKwargOverWrites(unittest.TestCase):

    def check_attributes(self, obj):
        self.assertTrue(hasattr(obj, 'a'))
        self.assertEqual('foo', obj.a)
        self.assertTrue(hasattr(obj, 'child'))
        self.assertIsInstance(obj.child, Child)
        self.assertTrue(hasattr(obj.child, 'b'))
        self.assertIsInstance(obj.child.b, float)
        self.assertEqual(42.0, obj.child.b)
        self.assertTrue(hasattr(obj.child, 'c'))
        self.assertIsInstance(obj.child.c, int)
        self.assertEqual(1, obj.child.c)

    def test_none(self):
        p = Parent({'child': {'c': 1}}, child=None)
        self.check_attributes(p)

    def test_dict(self):
        p = Parent({'child': {'c': 2}}, child={'c': 1})
        self.check_attributes(p)

    def test_self(self):
        p = Parent({'child': {'c': 2}}, child=Child({'c': 1}))
        self.check_attributes(p)

    def test_series(self):
        p = Parent({'child': {'c': 2}}, child=pd.Series({'c': 1}))
        self.check_attributes(p)

    def test_str(self):
        p = Parent({'child': {'c': 2}}, child="{'c': 1}")
        self.check_attributes(p)

    def test_json(self):
        p = Parent({'child': {'c': 2}}, child='{"c": 1}')
        self.check_attributes(p)

    def test_bytes(self):
        p = Parent({'child': {'c': 2}}, child=bytes('{"c": 1}'.encode('utf-8')))
        self.check_attributes(p)

    def test_bytearray(self):
        p = Parent(
            {'child': {'c': 2}},
            child=bytearray('{"c": 1}'.encode('utf-8'))
        )
        self.check_attributes(p)


class TestViews(unittest.TestCase):

    def setUp(self):
        self.p = Parent({'child': {'c': 1}})

    def test_str(self):
        expected = '{"a": "foo", "child": {"b": 42.0, "c": 1}}'
        self.assertEqual(expected, str(self.p))

    def test_repr(self):
        expected = (
            '{\n'
            '    "a": "foo",\n'
            '    "child": {\n'
            '        "b": 42.0,\n'
            '        "c": 1\n'
            '    }\n'
            '}'
        )
        self.assertEqual(expected, repr(self.p))

    def test_as_json(self):
        expected = {'a': 'foo', 'child': {'b': 42.0, 'c': 1}}
        self.assertDictEqual(expected, self.p.as_json)

    def test_as_dtype(self):
        expected = '{"a": "foo", "child": {"b": 42.0, "c": 1}}'
        self.assertEqual(expected, self.p.as_dtype)

    def test_as_series(self):
        expected = pd.Series(
            {'a': 'foo', 'child': '{"b": 42.0, "c": 1}'},
            name='Parent'
        )
        pd.testing.assert_series_equal(expected, self.p.as_series)


if __name__ == '__main__':
    unittest.main()

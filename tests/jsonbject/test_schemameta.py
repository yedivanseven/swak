import unittest
from unittest.mock import Mock
from swak.jsonobject.jsonobject import SchemaMeta
from swak.jsonobject.exceptions import SchemaError, DefaultsError
from swak.jsonobject.fields import Maybe


class TestEmptyAttributes(unittest.TestCase):

    def test_instantiation(self):

        class Empty(metaclass=SchemaMeta):
            pass

    def test_has_schema(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, 'schema'))

    def test_schema(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertDictEqual({}, Empty.schema)

    def test_has_defaults(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, 'defaults'))

    def test_defaults(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertDictEqual({}, Empty.defaults)


class TestAttributes(unittest.TestCase):

    def test_has_schema(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str

        self.assertTrue(hasattr(Test, 'schema'))

    def test_schema(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str

        self.assertDictEqual({'a': int, 'b': str}, Test.schema)

    def test_has_mixed_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str = 'hello'

        self.assertTrue(hasattr(Test, 'schema'))
        self.assertDictEqual({'a': int, 'b': str}, Test.schema)
        self.assertTrue(hasattr(Test, 'defaults'))
        self.assertDictEqual({'b': 'hello'}, Test.defaults)

    def test_mixed_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int
            b: str = 'hello'

        self.assertTrue(hasattr(Test, 'schema'))
        self.assertDictEqual({'a': int, 'b': str}, Test.schema)
        self.assertDictEqual({'b': 'hello'}, Test.defaults)

    def test_has_full_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'

        self.assertTrue(hasattr(Test, 'schema'))
        self.assertDictEqual({'a': int, 'b': str}, Test.schema)
        self.assertTrue(hasattr(Test, 'defaults'))

    def test_full_defaults(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'

        self.assertTrue(hasattr(Test, 'schema'))
        self.assertDictEqual({'a': int, 'b': str}, Test.schema)
        self.assertDictEqual({'a': 1, 'b': 'hello'}, Test.defaults)

    def test_caster_called(self):

        mock = Mock()

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: mock = '2.0'

        mock.assert_called_once()
        mock.assert_called_once_with('2.0')

    def test_cast_defaults_type(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: float = '2.0'

        self.assertIsInstance(Test.defaults['b'], float)

    def test_cast_defaults_value(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1.0
            b: float = '2.0'

        self.assertEqual(2.0, Test.defaults['b'])

    def test_has_class_variable(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertTrue(hasattr(Test, 'c'))

    def test_class_variable_type(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertIsInstance(Test.c, float)

    def test_class_variable_value(self):

        class Test(metaclass=SchemaMeta):
            a: int = 1
            b: str = 'hello'
            c = 2.0

        self.assertEqual(2.0, Test.c)


class TestValidations(unittest.TestCase):

    def test_annotation_not_callable(self):

        with self.assertRaises(SchemaError) as error:

            class Test(metaclass=SchemaMeta):
                a: int
                b: 1

        expected = '\nAll schema annotations must be callable!'
        self.assertEqual(expected, str(error.exception))

    def test_field_blacklisted(self):

        with self.assertRaises(SchemaError):

            class Test(metaclass=SchemaMeta):
                a: int
                schema: str

    def test_field_double_underscore(self):

        with self.assertRaises(SchemaError) as error:

            class Test(metaclass=SchemaMeta):
                a: int
                __b: str

        expected = '\nField names must not start with "__"!'
        self.assertEqual(expected, str(error.exception))

    def test_combi_message(self):

        with self.assertRaises(SchemaError) as error:

            class Test(metaclass=SchemaMeta):
                a: int
                b: 1
                __c: str

        expected = ('\nAll schema annotations must be callable!'
                    '\nField names must not start with "__"!')
        self.assertEqual(expected, str(error.exception))

    def test_default_none(self):

        with self.assertRaises(DefaultsError) as error:

            class Test(metaclass=SchemaMeta):
                a: int
                b: str = None
        expected = ("\nFor defaults ['b'] to be None, mark them"
                    " as Maybe(<YOUR_TYPE>) in the schema!")
        self.assertEqual(expected, str(error.exception))

    def test_default_wrong_type(self):

        with self.assertRaises(DefaultsError) as error:

            class Test(metaclass=SchemaMeta):
                a: str
                b: int = '1.0'

        expected = "\nDefaults ['b'] can not be cast to the desired types!"
        self.assertEqual(expected, str(error.exception))


class TestMaybe(unittest.TestCase):

    def test_maybe_works_with_none(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = None
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.defaults['a'])

    def test_maybe_works_with_value(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 1.0
            b: str = 'foo'
            c: float

        self.assertIsInstance(Test.defaults['a'], int)
        self.assertEqual(1, Test.defaults['a'])

    def test_maybe_works_with_null_str(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 'null'
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.defaults['a'])

    def test_maybe_works_with_none_str(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int) = 'None'
            b: str = 'foo'
            c: float

        self.assertIsNone(Test.defaults['a'])

    def test_maybe_works_without_default(self):

        class Test(metaclass=SchemaMeta):
            a: Maybe[int](int)
            b: str = 'foo'
            c: float

        self.assertDictEqual({'b': 'foo'}, Test.defaults)

    def test_maybe_lets_cast_error_through(self):

        with self.assertRaises(DefaultsError) as error:
            class Test(metaclass=SchemaMeta):
                a: Maybe[int](int) = '1.0'
                b: str = 'foo'
                c: float

        expected = "\nDefaults ['a'] can not be cast to the desired types!"
        self.assertEqual(expected, str(error.exception))


class TestExtraNone(unittest.TestCase):

    def test_has_ignore_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, 'ignore_extra'))

    def test_ignore_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertFalse(Empty.ignore_extra)

    def test_has_raise_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, 'raise_extra'))

    def test_raise_extra(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(Empty.raise_extra)

    def test_has_respect_none(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertTrue(hasattr(Empty, 'respect_none'))

    def test_respect_none(self):

        class Empty(metaclass=SchemaMeta):
            pass

        self.assertFalse(Empty.respect_none)


class TestKwargFields(unittest.TestCase):

    def test_add_kwarg_field_string(self):

        class Test(metaclass=SchemaMeta, foo='bar'):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float, 'foo': str}
        expected_defaults = {'a': 1, 'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_add_kwarg_field_float(self):

        class Test(metaclass=SchemaMeta, foo=1.0):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float, 'foo': str}
        expected_defaults = {'a': 1, 'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_body_field_overwrites_kwarg_field_and_default(self):

        class Test(metaclass=SchemaMeta, b=True):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_body_default_overwrites_kwarg_default(self):

        class Test(metaclass=SchemaMeta, a=True):
            a: int = 1
            b: float

        expected_schema = {'a': int, 'b': float}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)


class TestInheritance(unittest.TestCase):

    class A(metaclass=SchemaMeta):
        a: int

    class Adef(metaclass=SchemaMeta):
        a: int = 1

    def test_add_field(self):

        class AddB(self.A):
            b: str

        self.assertIsInstance(AddB, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str}, AddB.schema)
        self.assertDictEqual({}, AddB.defaults)

    def test_add_new_default_field(self):

        class AddBdef(self.A):
            b: str = 'hello'

        self.assertIsInstance(AddBdef, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str}, AddBdef.schema)
        self.assertDictEqual({'b': 'hello'}, AddBdef.defaults)

    def test_add_default_to_existing_field(self):

        class AddAdef(self.A):
            a: int = 2

        self.assertIsInstance(AddAdef, SchemaMeta)
        self.assertDictEqual({'a': int}, AddAdef.schema)
        self.assertDictEqual({'a': 2}, AddAdef.defaults)

    def test_change_default_of_existing_field(self):

        class ChangeAdef(self.Adef):
            a: int = 2

        self.assertIsInstance(ChangeAdef, SchemaMeta)
        self.assertDictEqual({'a': int}, ChangeAdef.schema)
        self.assertDictEqual({'a': 2}, ChangeAdef.defaults)

    def test_cant_remove_default_of_existing_field(self):

        class AundefA(self.Adef):
            a: int

        self.assertIsInstance(AundefA, SchemaMeta)
        self.assertDictEqual({'a': int}, AundefA.schema)
        self.assertDictEqual({'a': 1}, AundefA.defaults)

    def test_change_type_of_existing_field(self):

        class AchangeA(self.A):
            a: str

        self.assertIsInstance(AchangeA, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeA.schema)
        self.assertDictEqual({}, AchangeA.defaults)

    def test_change_type_of_existing_field_and_set_default(self):

        class AchangeAdef(self.A):
            a: str = 'foo'

        self.assertIsInstance(AchangeAdef, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeAdef.schema)
        self.assertDictEqual({'a': 'foo'}, AchangeAdef.defaults)

    def test_change_type_of_existing_field_and_change_default(self):

        class AchangeAdef(self.Adef):
            a: str = 'foo'

        self.assertIsInstance(AchangeAdef, SchemaMeta)
        self.assertDictEqual({'a': str}, AchangeAdef.schema)
        self.assertDictEqual({'a': 'foo'}, AchangeAdef.defaults)

    def test_grandparent(self):

        class B(self.A):
            b: str = 'foo'

        class C(B):
            c: float = 2.0

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str, 'c': float}, C.schema)
        self.assertDictEqual({'b': 'foo', 'c': 2.0}, C.defaults)

    def test_multiple_inheritance(self):

        class B(metaclass=SchemaMeta):
            b: str = 'foo'

        class C(self.A, B):
            c: float = 2.0

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual({'a': int, 'b': str, 'c': float}, C.schema)
        self.assertDictEqual({'b': 'foo', 'c': 2.0}, C.defaults)

    def test_mro(self):

        class B(metaclass=SchemaMeta):
            a: str

        class C(self.A, B):
            c: float

        class D(B, self.A):
            c: float

        self.assertIsInstance(C, SchemaMeta)
        self.assertDictEqual({'a': int, 'c': float}, C.schema)
        self.assertDictEqual({}, C.defaults)
        self.assertDictEqual({'a': str, 'c': float}, D.schema)
        self.assertDictEqual({}, D.defaults)

    def test_has_ignore_extra(self):

        class IgnoreExtra(self.A, ignore_extra=True):
            pass

        self.assertTrue(hasattr(IgnoreExtra, 'ignore_extra'))

    def test_ignore_extra(self):

        class IgnoreExtra(self.A, ignore_extra=True):
            pass

        self.assertTrue(IgnoreExtra.ignore_extra)

    def test_has_raise_extra(self):

        class RaiseExtra(self.A, raise_extra=False):
            pass

        self.assertTrue(hasattr(RaiseExtra, 'raise_extra'))

    def test_raise_extra(self):

        class RaiseExtra(self.A, raise_extra=False):
            pass

        self.assertFalse(RaiseExtra.raise_extra)

    def test_has_respect_none(self):

        class RespectNone(self.A, respect_none=True):
            pass

        self.assertTrue(hasattr(RespectNone, 'respect_none'))

    def test_respect_none(self):

        class RespectNone(self.A, respect_none=True):
            pass

        self.assertTrue(RespectNone.respect_none)

    def test_add_kwarg_field_string(self):

        class Test(self.A, foo='bar'):
            pass

        expected_schema = {'a': int, 'foo': str}
        expected_defaults = {'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_add_kwarg_field_float(self):

        class Test(self.A, foo=1.0):
            pass

        expected_schema = {'a': int, 'foo': str}
        expected_defaults = {'foo': 'foo'}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_body_field_overwrites_kwarg_field_and_default(self):

        class Test(self.A, a=True):
            pass

        expected_schema = {'a': int}
        expected_defaults = {}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)

    def test_body_default_overwrites_kwarg_default(self):

        class Test(self.Adef, a=True):
            a: int = 1

        expected_schema = {'a': int}
        expected_defaults = {'a': 1}
        self.assertDictEqual(expected_schema, Test.schema)
        self.assertDictEqual(expected_defaults, Test.defaults)


if __name__ == '__main__':
    unittest.main()
